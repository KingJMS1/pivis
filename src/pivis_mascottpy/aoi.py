import pandas as pd
from typing import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import figure, axes, contour, collections, animation
from .track import Track
from scipy import stats, special
import warnings

class AOIS:
    """
    Represents the various AOIs in an image as ovals (via Multivariate normal distribution)
    
    Attributes
    -----------
    mus : np.ndarray
        Centers of AOIS 
    covs : np.ndarray
        Covariance matrices for AOIS
    mvns : List[stats.rv_continuous]
        List of multivariate normal distributions corresponding to mus/covs.
    track: Track
        Instance of Track, containing data for plotting
    """
    def __init__(self, mus: np.ndarray, covs: np.ndarray, mvns: List[stats.rv_continuous], track: Track):
        self.mus = mus
        self.covs = covs
        self.mvns = mvns
        self.track = track

    def plot(self, method: Literal["draws", "ovals"] = "draws", figsize: Tuple[int] = (15, 10)) -> Tuple[figure.Figure, axes.Axes, List[contour.QuadContourSet | collections.PathCollection]]:
        """
        Plot the AOIs
        
        Parameters
        -----------
        method : Literal["draws", "ovals"]
            Choose whether to plot the AOIs as ovals or as multiple draws from their distribution
        figsize : Tuple[int]
            Size of figure to plot, in inches

        Returns
        --------
        Tuple[figure.Figure, axes.Axes, List[contour.QuadContourSet | collections.PathCollection]]
            Figure, axes, and list of matplotlib artists created
        """
    
        # Initialize image
        fig, ax = plt.subplots()
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        plt.imshow(self.track.img)
        plots = []
        img = self.track.img

        if method == "draws":
            draws = [x.rvs(1000) for x in self.mvns]
            for draw in draws:
                plots.append(plt.scatter(np.clip(draw[:, 0], 0, max(img.shape)), np.clip(draw[:, 1], 0, min(img.shape)), alpha=0.3))

        elif method == "ovals":
            xx = np.linspace(0, img.shape[1], img.shape[1])
            yy = np.linspace(0, img.shape[0], img.shape[1])
            x, y = np.meshgrid(xx, yy)
            ls = [self.mvns[i].pdf(np.array([x, y]).T) for i in range(len(self.mvns))]
            plots.append([ax.contour(xx, yy, ls[i].T, levels=[1e-7], alpha=1, origin="image") for i in range(len(ls))])

        else:
            raise ValueError(f"Invalid value for method in AOIs.plot, {method}, only 'draws' or 'ovals' is accepted.")
        
        return fig, ax, plots

    def video(self, figsize: Tuple[int] = (15, 10), fileloc = "vid.mp4", ops: int = 25) -> None:
        """
        Generates a video from the AOIs for a given track

        Parameters
        -----------
        xlab: str
            Label for X axis
        ylab: str
            Label for Y axis
        figsize: Tuple[int]
            Size for matplotlib figure in vidoe
        fileloc: str
            Location to save video
        ops: int
            Number of observations per second to show in the video, i.e. if period is 25 for the track, then at 40 ops, 1 second of video corresponds to 1 second of real time.
        """
        # Initialize variables
        data = self.track.raw_df
        img = self.track.img
        mvns = self.mvns
        FuncAnimation = animation.FuncAnimation

        time_col, IMG_X_col, IMG_Y_col, type_col, fixation_X_col, fixation_Y_col = self.track.cols
        xlab = IMG_X_col
        ylab = IMG_Y_col

        # Cluster points
        toCluster = data[[IMG_X_col, IMG_Y_col]].to_numpy()
        probs = special.softmax(np.array([x.logpdf(toCluster) for x in mvns]), axis=0).T
        cluster = np.argmax(probs, axis=1)

        # Function to animate each frame
        def plot_frame(frame):
            newPt = data.iloc[frame, ]
            scatter.set_offsets((newPt[xlab], newPt[ylab]))
            for contour in contours:
                contour.collections[0].set_alpha(0)
            contours[cluster[frame]].collections[0].set_alpha(1)
            return scatter, *[cont.collections[0] for cont in contours]
        
        # Spaces on which to calculate ovasl for AOIs
        xx = np.linspace(0, img.shape[1], img.shape[1])
        yy = np.linspace(0, img.shape[0], img.shape[1])
        x, y = np.meshgrid(xx, yy)
        ls = [mvns[i].pdf(np.array([x, y]).T) for i in range(len(mvns))]

        # Setup image
        fig, ax = plt.subplots()
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        ax.imshow(img)

        scatter = ax.scatter(data.iloc[0, ][xlab], data.iloc[0, ][ylab], marker="x", s=100, color="red")
        contours = [ax.contour(xx, yy, ls[i].T, levels=[1e-7], alpha=1, origin="image") for i in range(len(ls))]
        ani = FuncAnimation(fig, plot_frame, frames=range(len(cluster)), init_func=lambda: plot_frame(0), blit=True)
        ani.save(fileloc, fps=ops)


    @staticmethod
    def _kls(mus: np.ndarray, covs: np.ndarray) -> np.ndarray:
        """Calculate between group KL-divergences"""
        muq = mus.reshape(1, *mus.shape)
        mup = mus.reshape(mus.shape[0], 1, mus.shape[1])
        covq = covs.reshape(1, *covs.shape)
        covp = covs.reshape(covs.shape[0], 1, covs.shape[1], covs.shape[2])

        kl = -0.5 * np.log(np.linalg.det(covq) / np.linalg.det(covp)) - 0.5 * covs.shape[2] + 0.5 * np.trace(covq @ np.linalg.inv(covp), axis1=2, axis2=3) + 0.5 * ((muq - mup).reshape(mus.shape[0], mus.shape[0], 1, 2) @ np.linalg.solve(covp, (muq - mup).reshape(mus.shape[0], mus.shape[0], 2, 1))).squeeze()
        np.fill_diagonal(kl, 0)
        return kl

    @classmethod
    def from_track(cls, track: Track, threshold: float = 9.0, shortThreshold: float = 1000.0, verbose = True) -> Self:
        """
        Generates AOIs from tracking information

        Parameters
        -----------
        track : Track
            Tracking information to use
        threshold : float
            Threshold for KL-divergence used to combine multiple AOIs into larger AOIs, higher is more permissive.
        shortThreshold : float
            Threshold for (max timestamp - min timestamp) that this AOI is looked at, in milliseconds, AOI will be removed if below this threshold.
            Note: This is NOT a threshold for the cumulative amount of time that the AOI is looked at.
        verbose : bool
            Whether or not to print out the AOIs after each iteration
        """
        # Ignore warnings for some 'bad' things we will do
        warnings.filterwarnings(action='ignore')
        
        # Initialize variables
        mus = None
        covs = None

        data = track.df.copy()
        time_col, IMG_X_col, IMG_Y_col, type_col, fixation_X_col, fixation_Y_col = track.cols
        combined = True
        prevamt = 100000
        while combined:
            # Preprocessing to acquire data grouped by fixation point.
            grouped = data.groupby([fixation_X_col, fixation_Y_col])
            badPoints = (grouped.count()[time_col] == 1).to_numpy()
            means = grouped[[IMG_X_col, IMG_Y_col]].mean()
            covs = grouped[[IMG_X_col, IMG_Y_col]].cov()
            mus = means.to_numpy()
            covs = covs.to_numpy().reshape((mus.shape[0], 2, 2))
            goodidx = ~(np.any(np.isnan(mus)) | badPoints)
            index = means.index[goodidx]
            covs = covs[goodidx] + np.repeat(np.eye(2).reshape((1, 2, 2)), np.sum(goodidx), axis=0)
            mus = mus[goodidx]

            # Combine clusters close enough to each other
            kl = AOIS._kls(mus, covs)
            groups = []
            groupmap = {}
            xdx, ydx = np.nonzero((kl < threshold) & (kl != 0))
            if verbose:
                print(xdx)
                print(ydx)
                print(len(index))
                print()
            if len(index) < prevamt:
                prevamt = len(index)
            else:
                combined = False
                break

            for x, y in zip(xdx, ydx):
                if (x in groupmap) and (y in groupmap):
                    pass
                elif (x not in groupmap) and (y not in groupmap):
                    groups.append(set([x, y]))
                    groupmap[x] = len(groups) - 1
                    groupmap[y] = len(groups) - 1
                elif x in groupmap:
                    groups[groupmap[x]].add(y)
                    groupmap[y] = groupmap[x]
                elif y in groupmap:
                    groups[groupmap[y]].add(x)
                    groupmap[x] = groupmap[y]
            
            for i in range(data.shape[0]):
                x, y = data.iloc[i][[fixation_X_col, fixation_Y_col]]
                if not np.isnan(x):
                    origLoc = np.nonzero(index == (x, y))[0]
                    if len(origLoc) > 0:
                        origLoc = origLoc[0]
                        if goodidx[origLoc]:
                            if origLoc in groupmap:
                                newx, newy = index[min(groups[groupmap[origLoc]])]
                                data.iloc[i, data.columns.get_loc(fixation_X_col)] = newx
                                data.iloc[i, data.columns.get_loc(fixation_Y_col)] = newy
        
        # Revert warnings
        warnings.filterwarnings(action='once')
        
        # Remove too short AOIs
        shorts = (((grouped[time_col].max() - grouped[time_col].min())[goodidx]) < 1000).to_numpy()
        mus = mus[~shorts]
        covs = covs[~shorts]

        # Return AOIS
        mvns = [stats.multivariate_normal(mus[i], covs[i]) for i in range(mus.shape[0])]
        return cls(mus, covs, mvns, track)