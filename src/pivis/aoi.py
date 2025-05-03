import pandas as pd
from typing import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import figure, axes, contour, collections, animation
from .track import Track
from scipy import stats, special
import warnings

class AreasOfInterest:
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
    track : Track
        Instance of Track, containing data for plotting
    no_cluster : np.ndarray
        Boolean array indicating if the raw data is clustered properly (i.e. eye tracker found eyes) at the last 
        cluster_len observations.
    cluster : np.ndarray
        Raw data, clustered by last cluster_len observations, corresponds to indices in mvns/mus/covs.
    transition_probs : np.ndarray
        Probability of transitioning from 1 cluster to another
    """
    
    def __init__(self, mus: np.ndarray, covs: np.ndarray, mvns: List[stats.rv_continuous], track: Track, cluster_len: int, no_cluster: np.ndarray, cluster: np.ndarray, transition_probs: np.ndarray):
        self.mus = mus
        self.covs = covs
        self.mvns = mvns
        self.track = track
        self.cluster_len = cluster
        self.no_cluster = no_cluster
        self.cluster = cluster
        self.transition_probs = transition_probs

    def plot(self, method: Literal["draws", "ovals"] = "draws", figsize: Tuple[int] = (15, 10), plot_transitions: bool = False, transitionThreshold: float = 0.3) -> Tuple[figure.Figure, axes.Axes, List[contour.QuadContourSet | collections.PathCollection]]:
        """
        Plot the AOIs, Plot is left in memory so the user can plot extra things on it/change the title, etc.
        
        Parameters
        -----------
        method : Literal["draws", "ovals"]
            Choose whether to plot the AOIs as ovals or as multiple draws from their distribution
            Default is "draws"
        figsize : Tuple[int]
            Size of figure to plot, in inches
            Default is (15, 10)
        plot_transitions: bool
            Whether or not to plot common transitions
            Default is False.
        transitionThreshold: float
            Threshold for probability of a transition to occur, above which the transition will be plotted.
            Default is 0.3
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

        # Draw clusters via the chosen method
        if method == "draws":
            draws = (x.rvs(1000) for x in self.mvns)
            for draw in draws:
                plots.append(plt.scatter(np.clip(draw[:, 0], 0, max(img.shape)), np.clip(draw[:, 1], 0, min(img.shape)), alpha=0.3))

        elif method == "ovals":
            xx = np.linspace(0, img.shape[1], img.shape[1])
            yy = np.linspace(0, img.shape[0], img.shape[1])
            colorlist = ["green", "red", "purple", "black", "orange", "blue", "yellow", "pink", "lightgreen", "brown"]
            x, y = np.meshgrid(xx, yy)
            for i in range(len(self.mvns)):
                lik = self.mvns[i].pdf(np.array([x, y]).T)
                plots.append(ax.contour(xx, yy, lik.T, levels=[1e-7], alpha=1, origin="image", colors=colorlist[i % len(colorlist)]))

        else:
            raise ValueError(f"Invalid value for method in AOIs.plot, {method}, only 'draws' or 'ovals' is accepted.")
        
        if plot_transitions:
            for i, transition in enumerate(self.transition_probs):
                for j, prob in enumerate(transition):
                    if prob > transitionThreshold:
                        x = [self.mus[i][0], self.mus[j][0]]
                        y = [self.mus[i][1], self.mus[j][1]]
                        plt.plot(x, y, color="black", alpha=prob)
                        slope = (y[1] - y[0]) / (x[1] - x[0])
                        p25Dist = 0.25 * (x[1] - x[0])
                        newy = y[0] + p25Dist * slope
                        newx = p25Dist + x[0]
                        plt.arrow(newx, newy, np.sign(x[1] - x[0]), np.sign(x[1] - x[0]) * (y[1] - y[0]) / (x[1] - x[0]), head_width=40)

        return fig, ax, plots

    def video(self, figsize: Tuple[int] = (15, 10), fileloc: str = "vid.mp4", ops: int = 25, show_obs: int = 5, verbose: bool = False) -> None:
        """
        Generates a video from the AOIs for a given track

        Parameters
        -----------
        figsize: Tuple[int]
            Size for matplotlib figure in vidoe
            Default is (15, 10)
        fileloc: str
            Location to save video
            Default is "vid.mp4"
        ops: int
            Number of observations per second to show in the video, i.e. if period is 25 for the track, then at 40 ops, 1 second of video corresponds to 1 second of real time.
            Default is 25
        show_obs: int
            Number of observations on screen in any 1 frame.
            Default is 5
        verbose: bool
            Whether or not to print progress, prints every 50 frames.
        """
        # Initialize variables
        data = self.track.raw_df
        img = self.track.img
        mvns = self.mvns
        cluster = self.cluster
        bads = self.no_cluster
        FuncAnimation = animation.FuncAnimation
        time_col, IMG_X_col, IMG_Y_col, type_col, fixation_X_col, fixation_Y_col = self.track.cols
        xlab = IMG_X_col
        ylab = IMG_Y_col
        
        # Function to animate each frame
        def plot_frame(frame):
            if verbose:
                if frame % 50 == 0:
                    print(frame)
            newPt = data.iloc[frame:(frame + show_obs), ]
            scatter.set_offsets(np.array([newPt[xlab], newPt[ylab]]).T)
            if frame == 0:
                for contour in contours:
                    contour.set_alpha(0)
                contours[cluster[frame]].set_alpha(1)
            else:
                contours[cluster[frame - 1]].set_alpha(0)
                if not bads[frame]:
                    contours[cluster[frame]].set_alpha(1)
            return scatter, *[cont for cont in contours]
        
        # Spaces on which to calculate ovals for AOIs
        xx = np.linspace(0, img.shape[1], img.shape[1])
        yy = np.linspace(0, img.shape[0], img.shape[1])
        x, y = np.meshgrid(xx, yy)
        ls = (mvn.pdf(np.array([x, y]).T) for mvn in mvns)

        # Setup image
        fig, ax = plt.subplots()
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        ax.imshow(img)

        # Do the animation
        scatter = ax.scatter(data.iloc[0:show_obs, ][xlab], data.iloc[0:show_obs, ][ylab], marker="x", s=100, color="red")
        contours = [ax.contour(xx, yy, val.T, levels=[1e-7], alpha=1, origin="image") for val in ls]
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
    def from_track(cls, track: Track, threshold: float = 7.0, group_lim = 3, det_lim = 1e7, shortThreshold: int = 500, verbose: bool = True, cluster_len: int = 5) -> Self:
        """
        Generates AOIs from tracking information

        Parameters
        -----------
        track : Track
            Tracking information to use
        threshold : float
            Threshold for KL-divergence used to combine multiple AOIs into larger AOIs, higher is more permissive.
            Default is 7
        group_lim : int
            Controls how many clusters can be rolled into 1 larger cluster in each stage of the algorithm. Higher numbers create more large groups.
            Default is 5.
        det_lim : float
            Controls the maximum size of clusters via the determinant of their covariance matrices. Larger = larger clusters.
            Default is 1e7
        shortThreshold : int 
            Threshold for (max timestamp - min timestamp) that this AOI is looked at, in milliseconds, AOI will be removed if below this threshold.
            Note: This is NOT a threshold for the cumulative amount of time that the AOI is looked at.
            Default is 500
        verbose : bool
            Whether or not to print out the number of AOIs remaining after each iteration.
            Default is False.
        cluster_len : int
            After generating clusters based on a cleaned version of the data, 
            when looking at the raw data, how many previous observations should be used to determine the cluster of the current observation.
            (i.e. a smoothing factor for clustering the raw data, higher is smoother)
            Default is 5
        """
        # Ignore warnings for some 'bad' things we will do
        warnings.filterwarnings(action='ignore')
        
        # Initialize variables
        mus = None
        covs = None

        data = track.df.copy()
        time_col, IMG_X_col, IMG_Y_col, type_col, fixation_X_col, fixation_Y_col = track.cols
        combined = True
        prevamt = 1e99
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

            # Used to reject if we are above the determinant limit
            above_det_lim = lambda allFXs, allFYs: np.linalg.det(data[(data[fixation_X_col].isin(allFXs)) & (data[fixation_Y_col].isin(allFYs))][[IMG_X_col, IMG_Y_col]].cov().to_numpy()) > det_lim

            # Combine clusters close enough to each other
            kl = AreasOfInterest._kls(mus, covs)
            groups = []
            groupmap = {}
            xdx, ydx = np.nonzero((kl < threshold) & (kl != 0))
            if verbose:
                print(len(index))
            if len(index) < prevamt:
                prevamt = len(index)
            else:
                combined = False
                break

            for x, y in zip(xdx, ydx):
                if (x in groupmap) and (y in groupmap):
                    pass
                elif x in groupmap:
                    if len(groups[groupmap[x]]) < group_lim:
                        allFXs = [index[y][0]] + [index[z][0] for z in groups[groupmap[x]]]
                        allFYs = [index[y][1]] + [index[z][1] for z in groups[groupmap[x]]]
                        if above_det_lim(allFXs, allFYs):
                            continue  
                        groups[groupmap[x]].add(y)
                        groupmap[y] = groupmap[x]
                elif y in groupmap:
                    if len(groups[groupmap[y]]) < group_lim:
                        allFXs = [index[x][0]] + [index[z][0] for z in groups[groupmap[y]]]
                        allFYs = [index[x][1]] + [index[z][1] for z in groups[groupmap[y]]]
                        if above_det_lim(allFXs, allFYs):
                            continue
                        groups[groupmap[y]].add(x)
                        groupmap[x] = groupmap[y]
                elif (x not in groupmap) and (y not in groupmap):
                    allFXs = [index[x][0], index[y][0]]
                    allFYs = [index[x][1], index[y][1]]
                    if above_det_lim(allFXs, allFYs):
                        continue
                    groups.append(set([x, y]))
                    groupmap[x] = len(groups) - 1
                    groupmap[y] = len(groups) - 1
            
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
        shorts = (((grouped[time_col].max() - grouped[time_col].min())[goodidx]) < shortThreshold).to_numpy()
        mus = mus[~shorts]
        covs = covs[~shorts]

        # Cluster points, first generate clusters based on results
        # from cleaned data
        data = track.raw_df
        mvns = [stats.multivariate_normal(mus[i], covs[i]) for i in range(mus.shape[0])]

        # Use the last cluster_len observations to cluster
        toCluster = data[[IMG_X_col, IMG_Y_col]].to_numpy()[:, :, None].repeat(cluster_len, axis=2)
        for i in range(toCluster.shape[2]):
            toCluster[:, :, i] = np.roll(toCluster[:, :, i], i, axis=0)

        # Cut off the first cluster_len - 1 observations
        toCluster = toCluster[(cluster_len - 1):]
        
        # Associate points to clusters based on maximum likelihood
        lpdfs = np.array([x.logpdf(np.moveaxis(toCluster, 1, 2)) for x in mvns])
        lpdfs = np.sum(lpdfs, axis=2)
        probs = special.softmax(lpdfs, axis=0).T
        noCluster = np.all(np.isnan(probs), axis=1)
        cluster = np.argmax(probs, axis=1)
        
        # Comptue transition probabilities
        cluster = np.argmax(probs, axis=1)
        clusterNoNa = cluster[~np.all(np.isnan(probs), axis=1)]
        transitions = np.zeros((mus.shape[0], mus.shape[0]))
        for i in range(1, clusterNoNa.shape[0]):
            prev = clusterNoNa[i - 1]
            curr = clusterNoNa[i]
            if prev != curr:
                transitions[prev, curr] += 1
        transitionProbs = transitions / np.sum(transitions, axis=1).reshape((transitions.shape[0], 1))
        
        return cls(mus, covs, mvns, track, cluster_len, noCluster, cluster, transitionProbs)