from typing import *
import warnings
import shutil
import itertools
import pickle
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import figure, axes, contour, collections, animation
from scipy import stats, special
from PIL import Image
import Levenshtein
import imageio_ffmpeg


from .track import Track

class UnsupervisedAreasOfInterest:
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
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        
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
            kl = UnsupervisedAreasOfInterest._kls(mus, covs)
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
    
class SupervisedAreasOfInterest:
    """
    Represents the various AOIs in an image as ovals (via Multivariate normal distribution)
    
    Attributes
    -----------
    aois : List[np.ndarray]
        List of all aois, stored as images indicating where they are in the overall image
    track : Track
        Instance of Track, containing data for plotting
    cluster_len : int
        Number of observations to use to smooth raw data before clustering into AOIs.
    transition_probs : np.ndarray
        Probability of transitioning from 1 cluster to another
    scan_string : np.ndarray
        List of clusters looked at, in chronological order
    colors : np.ndarray
        All AOI colors in the original image, after removing edge blending
    aoi_img : np.ndarray
        Image showing all the AOIs
    cluster_img : np.ndarray
        Image similar to aoi_img but with colors replaced with indices in colors
    """
    
    def __init__(self, aois: List[np.ndarray], track : Track, cluster_len : int, transition_probs : np.ndarray, scan_string : np.ndarray, colors : np.ndarray, aoi_img : np.ndarray, cluster_img: np.ndarray):
        self.aois = aois
        self.track = track
        self.cluster_len = cluster_len
        self.transition_probs = transition_probs
        self.scan_string = scan_string
        self.colors = colors
        self.aoi_img = aoi_img
        self.cluster_img = cluster_img

    def plot(self, figsize: Tuple[int] = (15, 10), plot_transitions: bool = False, transitionThreshold: float = 0.3) -> Tuple[figure.Figure, axes.Axes]:
        """
        Plot the AOIs, Plot is left in memory so the user can plot extra things on it/change the title, etc.
        
        Parameters
        -----------
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
        Tuple[figure.Figure, axes.Axes]
            Figure, axes of created image
        """
        # Initialize image
        fig, ax = plt.subplots()
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        plt.imshow(self.track.img)
        plots = []
        img = self.track.img

        # Draw clusters
        plt.imshow(self.aoi_img)

        if plot_transitions:
            for i, transition in enumerate(self.transition_probs):
                for j, prob in enumerate(transition):
                    if prob > transitionThreshold:
                        mu_i = np.mean(np.argwhere(self.cluster_img == i), axis=0)
                        mu_j = np.mean(np.argwhere(self.cluster_img == j), axis=0)
                        x = [mu_i[0], mu_j[0]]
                        y = [mu_i[1], mu_j[1]]
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
            return scatter,
        
        # Setup image
        fig, ax = plt.subplots()
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        ax.imshow(img)

        # Do the animation
        scatter = ax.scatter(data.iloc[0:show_obs, ][xlab], data.iloc[0:show_obs, ][ylab], marker="x", s=100, color="red")
        ani = FuncAnimation(fig, plot_frame, frames=range(data.shape[0]), init_func=lambda: plot_frame(0), blit=True)
        ani.save(fileloc, fps=ops)

    def save(self, fileloc: str = "out.pkl"):
        with open(fileloc, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(fileloc: str = "out.pkl") -> Self:
        with open(fileloc, "rb") as file:
            return pickle.load(file)
    
    @staticmethod
    def from_file(fileloc: str = "out.pkl") -> Self:
        return SupervisedAreasOfInterest.load(fileloc)

    @classmethod
    def from_labelled_img(cls, labelled_img: str, track: Track, verbose: bool = True, cluster_len: int = 5, figsize=(10,8), aoi_img_dir = "./aois", remove_background = False, first_run = False) -> Self:
        """
        Generates AOIs from tracking information

        Parameters
        -----------
        labelled_img: str
            Path to image with AOIS labelled as differently colored areas
        track : Track
            Tracking information to use
        verbose : bool
            Whether or not to print out the number of AOIs remaining after each iteration.
            Default is False.
        cluster_len : int
            When looking at the raw data, how many previous observations should be used to determine the cluster of the current observation.
            (i.e. a smoothing factor for clustering the raw data, higher is smoother)
            Default is 5
        figsize : Tuple[int]
            Specifies how large the images showing what each AOI corresponds to should be.
        aoi_img_dir : str
            Path to directory that images showing what each AOI corresponds to will be stored.
        remove_background : bool
            Whether or not to include the background AOI in the results
        """
        
        # Load in the image
        img = Image.open(labelled_img)
        arr = np.array(img)
        arr = arr[:, :, :3]

        # Find unique colors in the image
        flatImg = np.reshape(arr, (-1, 3))
        uniq, counts = np.unique(flatImg, axis=0, return_counts=True)

        # Try to ignore blending at edges in image
        badColors = counts < 200
        colors = uniq[~badColors]
        print("Dealing with blending")
        for i, badColor in enumerate(uniq[badColors]):
            if i % 100 == 0:
                print(i / len(uniq[badColors]))
            
            # Find the closest color in the good colors
            bestRepl = np.argmin(np.sum(np.square(colors - badColor), axis=1))

            # Find all places in image with bad color and replace with good color
            idxs = np.all(flatImg == badColor, axis=1)
            flatImg[idxs] = colors[bestRepl]

        # Make an 'image' consisting of indexes into the colors list.
        cluster_idx_img = np.zeros((flatImg.shape[0]))
        for i, color in enumerate(colors):
            idxs = np.all(flatImg == color, axis=1)
            cluster_idx_img[idxs] = i
        
        # Recompute the unique colors in the image, and put the image back together after removing
        # edge blending
        uniq, counts = np.unique(flatImg, axis=0, return_counts=True)
        reImg = np.reshape(flatImg, arr.shape)
        cluster_idx_img = np.reshape(cluster_idx_img, arr.shape[:-1])

        # Delete the old aois directory, replace with a new one
        if first_run:
            shutil.rmtree(aoi_img_dir, ignore_errors=True)
            os.makedirs(aoi_img_dir, exist_ok=True)
        
        aoiList = []
        # Write out the image files so someone can fix them if necessary, and for interpretation later
        for x in np.unique(cluster_idx_img):
            aoiList.append(cluster_idx_img == x)
            if first_run:
                fig, ax = plt.subplots(figsize=figsize)
                plt.imshow(aoiList[-1])
                plt.title(f"AOI {x}")
                plt.savefig(os.path.join(aoi_img_dir, f"{x}.png"))
                plt.close(fig)
        
        if first_run:
            print("Exiting due to first_run flag being set to true. Check the aois folder for all identified AOIs and correct the underlying image file to have solid blocks of color rather than feathered/blended edges on the color blocks.")
            exit(0)

        time_col, IMG_X_col, IMG_Y_col, type_col, fixation_X_col, fixation_Y_col = track.cols
        data = track.raw_df

        # Use the last cluster_len observations to cluster
        toCluster = data[[IMG_X_col, IMG_Y_col]].to_numpy()[:, :, None].repeat(cluster_len, axis=2)
        for i in range(toCluster.shape[2]):
            toCluster[:, :, i] = np.roll(toCluster[:, :, i], i, axis=0)

        # Cut off the first cluster_len - 1 observations
        toCluster = np.nan_to_num(toCluster[(cluster_len - 1):], nan=-1).astype("int64")
        
        # Remove places where eye tracker stopped tracking
        bad = np.any(np.any(toCluster == -1, axis=-1), axis=-1)

        # Cluster the track into the AOIs
        clustered = cluster_idx_img[toCluster[:, 1, :], toCluster[:, 0, :]]
        clustered = stats.mode(clustered, axis=1).mode
        clustered[bad] = -1

        background_idx = np.argmin(np.sum(np.square(uniq - np.array([255, 255, 255])), axis=1))
        print(f"AOI {background_idx} is assumed to be the background")

        # Remove background and duplicates 
        aoiStrs = np.array([key for key, group in itertools.groupby(clustered[~bad])])
        if remove_background:
            aoiStrs = aoiStrs[aoiStrs != background_idx] # Ignore cluster indicating the background
            aoiStrs = np.array([key for key, group in itertools.groupby(aoiStrs)]) # Deduplicate patterns once more

        # Compute transition probabilities between aois
        transitions = np.zeros((len(aoiList), len(aoiList)))
        for i in range(1, aoiStrs.shape[0]):
            prev = aoiStrs[i - 1]
            curr = aoiStrs[i]
            if prev != curr:
                transitions[int(prev), int(curr)] += 1
        transitionProbs = transitions / np.sum(transitions, axis=1).reshape((transitions.shape[0], 1))

        return cls(aoiList, track, cluster_len, transitionProbs, aoiStrs, colors, reImg, cluster_idx_img)
    
    def compute_patterns(self, max_length: int, lev_dists: Tuple[int], weights: Tuple[float] = None, normalize = "n_patterns"):
        """
        Attempts to find consistent scan patterns within a file

        Parameters
        -----------
        max_length : int
            Maximum length of patterns to look for
        lev_dists : Tuple[int]
            Tuple of size max_length - 1, chooses how many Levenshtein distance away we should count observations of a pattern
        weights : Tuple[float]
            Tuple of size max_length - 1, chooses how to weight each observation of an observation a given Levenshtein distance away for the final consistency score,
            defaults to the following weighting scheme: 0.6, 0.2, 0.1, 0.05, etc. Should sum to 1
        normalize : str
            "n_patterns", "time" or anything else for neither, chooses how to normalize patternScores, by number of scan patterns found, by amount of time in file, or by nothing.
        Returns
        -------
        patterns : List[Tuple[Tuple, np.ndarray]]
            List of all patterns, along with the number of observations of each scan pattern, entries for observations separate out 0-distance away, 1-distance away, etc. observations.
            Multiplied by weights and divided by number of patterns in file to determine importance
        patternScores : np.ndarray
            Importance scores of each scan pattern in patterns
        """
        # Verify function inputs conform to assumptions
        if len(lev_dists) != max_length - 1:
            raise ValueError(f"lev_dists must be of size {max_length - 1}")
        if weights is not None and len(weights) != max_length - 1:
            raise ValueError(f"weights must be of size {max_length - 1}")
        if round(sum(weights), 10) != 1:
            raise ValueError(f"Weights must sum to 1 instead of {round(sum(weights), 10)}.")
        
        # Find all observations of scan patterns of up to maxLevDist away
        existingPatterns = {}
        maxPatternSize = max_length
        maxLevDist = lev_dists
        for size in range(2, maxPatternSize + 1):
            for x in np.lib.stride_tricks.sliding_window_view(self.scan_string, size):
                x = tuple(x)
                if x in existingPatterns:
                    existingPatterns[x][0] += 1
                else:
                    existingPatterns[x] = [1] + [0] * max(maxLevDist)
                    for z in existingPatterns.keys():
                        if z != x:
                            ld = Levenshtein.distance(x, z)
                            if ld <= maxLevDist[size - 2]:
                                levCutoff = maxLevDist[size - 2] - ld
                                levsToAdd = existingPatterns[z][:(levCutoff + 1)]
                                for i, lev in enumerate(levsToAdd):
                                    existingPatterns[x][1 + i] += lev
                for z in existingPatterns.keys():
                    if z == x:
                        pass
                    else:
                        ld = Levenshtein.distance(x, z)
                        if ld <= maxLevDist[size - 2]:
                            existingPatterns[z][ld] += 1
        
        # Calculate importance scores
        importance = weights
        if importance is None:
            importance = np.array([0.6] + [0.2 * (0.5 ** x) for x in range(max(maxLevDist))])
        importance[-1] += 1 - np.sum(importance)
        patterns = [(x, np.array(existingPatterns[x])) for x in existingPatterns]
        patterns = sorted(patterns, key = lambda x: np.sum(importance * x[1]), reverse=True)
        patternScores = np.array([np.sum(importance * x[1]) for x in patterns])

        if normalize == "n_patterns":
            patternScores = patternScores / len(patternScores)
        elif normalize == "time":
            patternScores = patternScores / (self.track.df.shape[0] / 50)
        
        return patterns, patternScores