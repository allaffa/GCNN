import matplotlib.pyplot as plt
from itertools import chain
import time, pickle
import numpy as np
from math import sqrt, floor, ceil

plt.rcParams.update({"font.size": 18})


class Visualizer:
    """A class used for visualizing values in a scatter plot. There are two attributes: true_values and predicted_values that we want to see
    in a scatter plot. The ideal case is that the values will be positioned on a thin diagonal of the scatter plot.

    Methods
    -------
    add_test_values(true_values: [], predicted_values: [])
        Add the true and predicted values to the lists.
    create_scatter_plot()
        Create the scatter plot out of true and predicted values.
    """

    def __init__(self, model_with_config_name: str):
        self.true_values = []
        self.predicted_values = []
        self.model_with_config_name = model_with_config_name

    def add_test_values(self, true_values: [], predicted_values: []):
        """Append true and predicted values to existing lists.

        Parameters
        ----------
        true_values: []
            List of true values to append to existing one.
        predicted_values: []
            List of predicted values to append to existing one.
        """
        self.true_values.extend(true_values)
        self.predicted_values.extend(predicted_values)

    def __convert_to_list(self):
        """When called it performs flattening of a list because the values that are stored in true and predicted values lists are in
        the shape: [[1], [2], ...] and in order to visualize them in scatter plot they need to be in the shape: [1, 2, ...].
        """
        if len(self.true_values) != len(self.predicted_values):
            print("Length of true and predicted values array is not the same!")

        elif len(self.true_values[0]) > 1 or len(self.predicted_values[0]) > 1:
            print(
                "Values inside true or predicted values list can only be scalars and not array of points!"
            )

        self.true_values = list(chain.from_iterable(self.true_values))
        self.predicted_values = list(chain.from_iterable(self.predicted_values))

    def hist2d_contour(self, data1, data2):
        hist2d_pasr, xedge_pasr, yedge_pasr = np.histogram2d(
            np.hstack(data1), np.hstack(data2), bins=50
        )
        xcen_pasr = 0.5 * (xedge_pasr[0:-1] + xedge_pasr[1:])
        ycen_pasr = 0.5 * (yedge_pasr[0:-1] + yedge_pasr[1:])
        BCTY_pasr, BCTX_pasr = np.meshgrid(ycen_pasr, xcen_pasr)
        hist2d_pasr = hist2d_pasr / np.amax(hist2d_pasr)
        return BCTX_pasr, BCTY_pasr, hist2d_pasr

    def hist1d_err(self, data1, data2, weight=1.0):
        errabs = np.abs(np.hstack(data1) - np.hstack(data2)) * weight

        hist2d_pasr, xedge_pasr, yedge_pasr = np.histogram2d(
            np.hstack(data1), errabs, bins=50
        )
        xcen_pasr = 0.5 * (xedge_pasr[0:-1] + xedge_pasr[1:])
        ycen_pasr = 0.5 * (yedge_pasr[0:-1] + yedge_pasr[1:])
        hist2d_pasr = hist2d_pasr / np.amax(hist2d_pasr)
        mean1d_cond = np.dot(hist2d_pasr, ycen_pasr) / np.sum(hist2d_pasr, axis=1)
        return xcen_pasr, mean1d_cond

    def create_scatter_plot(self, ivar, save_plot=True):
        """Creates scatter plot from stored values in the tru and  predicted values lists."""
        # self.__convert_to_list()
        nshape = np.asarray(self.predicted_values).shape
        if nshape[1] == 1:
            fig = plt.figure(figsize=(15, 4.5))
            plt.subplots_adjust(
                left=0.08, bottom=0.15, right=0.95, top=0.925, wspace=0.35, hspace=0.1
            )
            ax = plt.subplot(1, 3, 1)
            plt.scatter(self.true_values, self.predicted_values)
            plt.title("Scalar output")
            plt.xlabel("True value")
            plt.ylabel("Predicted value")
            ax.set_aspect("equal")
            minimum = np.min((ax.get_xlim(), ax.get_ylim()))
            maximum = np.max((ax.get_xlim(), ax.get_ylim()))
            ax.set_xlim(minimum, maximum)
            ax.set_ylim(minimum, maximum)

            ax = plt.subplot(1, 3, 2)
            xtrue, error = self.hist1d_err(
                self.true_values, self.predicted_values, weight=1.0
            )
            plt.plot(xtrue, error, "ro")
            plt.title("Conditional mean abs error")
            plt.xlabel("True value")
            plt.ylabel("abs error")
            ax.set_xlim(minimum, maximum)

            ax = plt.subplot(1, 3, 3)
            hist1d, bin_edges = np.histogram(
                np.array(self.predicted_values) - np.array(self.true_values),
                bins=40,
                density=True,
            )
            plt.plot(0.5 * (bin_edges[:-1] + bin_edges[1:]), hist1d, "ro")
            plt.title("Scalar output: error PDF")
            plt.ylabel("PDF")

        else:
            fig = plt.figure(figsize=(18, 16))
            ax = plt.subplot(331)
            vlen_true = []
            vlen_pred = []
            vsum_true = []
            vsum_pred = []
            for isamp in range(nshape[0]):
                vlen_true.append(
                    sqrt(sum([comp ** 2 for comp in self.true_values[isamp][:]]))
                )
                vlen_pred.append(
                    sqrt(sum([comp ** 2 for comp in self.predicted_values[isamp][:]]))
                )
                vsum_true.append(sum(self.true_values[isamp][:]))
                vsum_pred.append(sum(self.predicted_values[isamp][:]))
            plt.scatter(vlen_true, vlen_pred)
            plt.title("Vector output: length")
            plt.xlabel("True value")
            plt.ylabel("Predicted value")
            ax.set_aspect("equal")
            minimum = np.min((ax.get_xlim(), ax.get_ylim()))
            maximum = np.max((ax.get_xlim(), ax.get_ylim()))
            ax.set_xlim(minimum, maximum)
            ax.set_ylim(minimum, maximum)

            ax = plt.subplot(334)
            xtrue, error = self.hist1d_err(
                vlen_true, vlen_pred, weight=1.0 / sqrt(nshape[1])
            )
            plt.plot(xtrue, error, "ro")
            plt.ylabel("Conditional mean abs error")
            plt.xlabel("True value")
            ax.set_xlim(minimum, maximum)

            ax = plt.subplot(337)
            hist1d, bin_edges = np.histogram(
                np.array(vlen_pred) - np.array(vlen_true), bins=40, density=True
            )
            plt.plot(0.5 * (bin_edges[:-1] + bin_edges[1:]), hist1d, "ro")
            plt.title("error PDF")
            plt.ylabel("PDF")
            ax.set_xlim(minimum, maximum)

            ax = plt.subplot(332)
            plt.scatter(vsum_true, vsum_pred)
            plt.title("Vector output: sum")
            plt.xlabel("True value")
            plt.ylabel("Predicted value")
            ax.set_aspect("equal")
            minimum = np.min((ax.get_xlim(), ax.get_ylim()))
            maximum = np.max((ax.get_xlim(), ax.get_ylim()))
            ax.set_xlim(minimum, maximum)
            ax.set_ylim(minimum, maximum)

            ax = plt.subplot(335)
            xtrue, error = self.hist1d_err(vsum_true, vsum_pred, weight=1.0 / nshape[1])
            plt.plot(xtrue, error, "ro")
            plt.xlabel("True value")
            ax.set_xlim(minimum, maximum)

            ax = plt.subplot(338)
            hist1d, bin_edges = np.histogram(
                np.array(vsum_pred) - np.array(vsum_true), bins=40, density=True
            )
            plt.plot(0.5 * (bin_edges[:-1] + bin_edges[1:]), hist1d, "ro")
            plt.title("error PDF")
            plt.ylabel("PDF")

            ax = plt.subplot(333)
            truecomp = []
            predcomp = []
            for icomp in range(nshape[1]):
                truecomp.append(self.true_values[:][icomp])
                predcomp.append(self.predicted_values[:][icomp])
                plt.scatter(self.true_values[:][icomp], self.predicted_values[:][icomp])
            plt.title("Vector output: components")
            plt.xlabel("True value")
            plt.ylabel("Predicted value")
            ax.set_aspect("equal")
            minimum = np.min((ax.get_xlim(), ax.get_ylim()))
            maximum = np.max((ax.get_xlim(), ax.get_ylim()))
            ax.set_xlim(minimum, maximum)
            ax.set_ylim(minimum, maximum)

            ax = plt.subplot(336)
            xtrue, error = self.hist1d_err(truecomp, predcomp, weight=1.0)
            plt.plot(xtrue, error, "ro")
            plt.xlabel("True value")
            ax.set_xlim(minimum, maximum)

            ax = plt.subplot(339)
            hist1d, bin_edges = np.histogram(
                np.array(predcomp) - np.array(truecomp), bins=40, density=True
            )
            plt.plot(0.5 * (bin_edges[:-1] + bin_edges[1:]), hist1d, "ro")
            plt.title("error")
            plt.ylabel("PDF")

            plt.subplots_adjust(
                left=0.075, bottom=0.1, right=0.98, top=0.95, wspace=0.2, hspace=0.25
            )

        if save_plot:
            fig.savefig(
                f"./logs/{self.model_with_config_name}/scatter_plot_test_ivar{ivar}.png"
            )
            plt.close()
        else:
            plt.show()

    def create_scatter_plot_atoms_hist(
        self, ivar, x_atomfeature, iepoch, save_plot=True
    ):
        """Creates scatter plot from stored values in the tru and  predicted values lists."""
        # self.__convert_to_list()
        nshape = np.asarray(self.predicted_values).shape
        varnames = ["free energy", "charge density", "magnetic moment"]
        if nshape[1] == 1:
            fig = plt.figure(figsize=(12, 6))
            ax = plt.subplot(1, 2, 1)
            plt.scatter(self.true_values, self.predicted_values)
            plt.title(varnames[ivar])
            plt.xlabel("True value")
            plt.ylabel("Predicted value")
            ax.set_aspect("equal")
            minimum = np.min((ax.get_xlim(), ax.get_ylim()))
            maximum = np.max((ax.get_xlim(), ax.get_ylim()))
            ax.set_xlim(minimum, maximum)
            ax.set_ylim(minimum, maximum)

            ax = plt.subplot(1, 2, 2)
            hist1d, bin_edges = np.histogram(
                np.array(self.predicted_values) - np.array(self.true_values),
                bins=40,
                density=True,
            )
            plt.plot(0.5 * (bin_edges[:-1] + bin_edges[1:]), hist1d, "ro")

            plt.title("Scalar output: error")
            plt.title(varnames[ivar] + ": error")

            plt.subplots_adjust(
                left=0.075, bottom=0.1, right=0.98, top=0.9, wspace=0.2, hspace=0.25
            )

            if save_plot:
                fig.savefig(
                    f"./logs/{self.model_with_config_name}/task{ivar}_"
                    + str(iepoch).zfill(4)
                    + ".png"
                )
                plt.close()
            return
        else:
            nrow = floor(sqrt((nshape[1] + 1)))
            ncol = ceil((nshape[1] + 1) / nrow)
            fig = plt.figure(figsize=(ncol * 4, nrow * 3.2))
            for iatom in range(nshape[1]):
                ax = plt.subplot(nrow, ncol, iatom + 1)
                xfeature = []
                truecomp = []
                predcomp = []
                smplordr = []
                for isamp in range(nshape[0]):
                    xfeature.append(x_atomfeature[isamp][iatom])
                    smplordr.append(x_atomfeature[isamp][iatom] * 0 + isamp)
                    truecomp.append(self.true_values[isamp][iatom])
                    predcomp.append(self.predicted_values[isamp][iatom])

                plt.scatter(truecomp, predcomp, 6, xfeature)

                plt.title("atom:" + str(iatom))
                ax.set_aspect("equal")
                # plt.colorbar()
                minimum = np.min((ax.get_xlim(), ax.get_ylim()))
                maximum = np.max((ax.get_xlim(), ax.get_ylim()))
                ax.set_xlim(minimum, maximum)
                ax.set_ylim(minimum, maximum)

            ax = plt.subplot(nrow, ncol, nshape[1] + 1)
            xfeature = []
            truecomp = []
            predcomp = []
            smplordr = []
            for isamp in range(nshape[0]):
                xfeature.append(sum(x_atomfeature[isamp][:]))
                smplordr.append(isamp)
                truecomp.append(sum(self.true_values[isamp][:]))
                predcomp.append(sum(self.predicted_values[isamp][:]))
            plt.scatter(truecomp, predcomp, 6, xfeature)
            plt.title("SUM")
            ax.set_aspect("equal")
            # plt.colorbar()
            minimum = np.min((ax.get_xlim(), ax.get_ylim()))
            maximum = np.max((ax.get_xlim(), ax.get_ylim()))
            ax.set_xlim(minimum, maximum)
            ax.set_ylim(minimum, maximum)

            plt.subplots_adjust(
                left=0.075, bottom=0.1, right=0.98, top=0.9, wspace=0.2, hspace=0.25
            )

            if save_plot:
                fig.savefig(
                    f"./logs/{self.model_with_config_name}/task{ivar}_atoms_"
                    + str(iepoch).zfill(4)
                    + ".png"
                )
                plt.close()

            nrow = floor(sqrt((nshape[1] + 1)))
            ncol = ceil((nshape[1] + 1) / nrow)
            fig = plt.figure(figsize=(ncol * 4, nrow * 3.2))
            for iatom in range(nshape[1]):
                ax = plt.subplot(nrow, ncol, iatom + 1)
                xfeature = []
                truecomp = []
                predcomp = []
                smplordr = []
                for isamp in range(nshape[0]):
                    xfeature.append(x_atomfeature[isamp][iatom])
                    smplordr.append(x_atomfeature[isamp][iatom] * 0 + isamp)
                    truecomp.append(self.true_values[isamp][iatom])
                    predcomp.append(self.predicted_values[isamp][iatom])

                hist1d, bin_edges = np.histogram(
                    np.array(predcomp) - np.array(truecomp), bins=40, density=True
                )
                plt.plot(0.5 * (bin_edges[:-1] + bin_edges[1:]), hist1d, "ro")
                plt.title("atom:" + str(iatom))

            ax = plt.subplot(nrow, ncol, nshape[1] + 1)
            xfeature = []
            truecomp = []
            predcomp = []
            smplordr = []

            for isamp in range(nshape[0]):
                xfeature.append(sum(x_atomfeature[isamp][:]))
                smplordr.append(isamp)
                truecomp.append(sum(self.true_values[isamp][:]))
                predcomp.append(sum(self.predicted_values[isamp][:]))
            hist1d, bin_edges = np.histogram(
                np.array(predcomp) - np.array(truecomp), bins=40, density=True
            )
            plt.plot(0.5 * (bin_edges[:-1] + bin_edges[1:]), hist1d, "ro")
            plt.title("SUM")

            plt.subplots_adjust(
                left=0.075, bottom=0.1, right=0.98, top=0.9, wspace=0.2, hspace=0.25
            )
            if save_plot:
                fig.savefig(
                    f"./logs/{self.model_with_config_name}/task{ivar}_error_"
                    + str(iepoch).zfill(4)
                    + ".png"
                )
                plt.close()

    def create_scatter_plot_atoms(self, ivar, x_atomfeature, save_plot=True):
        """Creates scatter plot from stored values in the tru and  predicted values lists."""
        # self.__convert_to_list()
        nshape = np.asarray(self.predicted_values).shape
        if nshape[1] == 1:
            return
        else:
            fig = plt.figure(figsize=(20, 20))
            nrow = 6
            ncol = 6
            for iatom in range(nshape[1]):
                ax = plt.subplot(nrow, ncol, iatom + 1)
                xfeature = []
                truecomp = []
                predcomp = []
                smplordr = []
                for isamp in range(nshape[0]):
                    xfeature.append(x_atomfeature[isamp][iatom])
                    smplordr.append(x_atomfeature[isamp][iatom] * 0 + isamp)
                    truecomp.append(self.true_values[isamp][iatom])
                    predcomp.append(self.predicted_values[isamp][iatom])

                plt.scatter(truecomp, predcomp, 6, xfeature)
                plt.title("atom:" + str(iatom))
                ax.set_aspect("equal")
                # plt.colorbar()
                minimum = np.min((ax.get_xlim(), ax.get_ylim()))
                maximum = np.max((ax.get_xlim(), ax.get_ylim()))
                ax.set_xlim(minimum, maximum)
                ax.set_ylim(minimum, maximum)

            ax = plt.subplot(nrow, ncol, nshape[1] + 1)
            xfeature = []
            truecomp = []
            predcomp = []
            smplordr = []
            for isamp in range(nshape[0]):
                xfeature.append(sum(x_atomfeature[isamp][:]))
                smplordr.append(isamp)
                truecomp.append(sum(self.true_values[isamp][:]))
                predcomp.append(sum(self.predicted_values[isamp][:]))
            plt.scatter(truecomp, predcomp, 60, xfeature)
            plt.title("SUM")
            ax.set_aspect("equal")
            # plt.colorbar()
            minimum = np.min((ax.get_xlim(), ax.get_ylim()))
            maximum = np.max((ax.get_xlim(), ax.get_ylim()))
            ax.set_xlim(minimum, maximum)
            ax.set_ylim(minimum, maximum)

            ax = plt.subplot(nrow, ncol, nshape[1] + 2)
            xfeature = []
            truecomp = []
            predcomp = []
            for iatom in range(nshape[1]):
                xfeature.append(sum(x_atomfeature[:][iatom]))
                truecomp.append(sum(self.true_values[:][iatom]))
                predcomp.append(sum(self.predicted_values[:][iatom]))

            plt.scatter(truecomp, predcomp, 60, xfeature)
            plt.title(
                "mean(atom:0-" + str(iatom) + ")"
            )  # +", "+str(sum(xfeature)/len(xfeature)))
            ax.set_aspect("equal")
            # plt.colorbar()
            minimum = np.min((ax.get_xlim(), ax.get_ylim()))
            maximum = np.max((ax.get_xlim(), ax.get_ylim()))
            ax.set_xlim(minimum, maximum)
            ax.set_ylim(minimum, maximum)

            plt.subplots_adjust(
                left=0.075, bottom=0.1, right=0.98, top=0.9, wspace=0.2, hspace=0.25
            )

            if save_plot:
                fig.savefig(
                    f"./logs/{self.model_with_config_name}/scatter_plot_test_ivar{ivar}_atoms_all.png"
                )
                plt.close()

    def plot_history(
        self,
        trainlib,
        vallib,
        testlib,
        tasklib,
        tasklib_vali,
        tasklib_test,
        tasklib_nodes,
        tasklib_vali_nodes,
        tasklib_test_nodes,
        task_weights,
    ):
        nrow = 1
        fhist = open(f"./logs/{self.model_with_config_name}/history_loss.pckl", "wb")
        pickle.dump(
            [
                trainlib,
                vallib,
                testlib,
                tasklib,
                tasklib_vali,
                tasklib_test,
                tasklib_nodes,
                tasklib_vali_nodes,
                tasklib_test_nodes,
                task_weights,
            ],
            fhist,
        )
        fhist.close()

        if len(tasklib) > 0:
            tasklib = np.array(tasklib)
            tasklib_vali = np.array(tasklib_vali)
            tasklib_test = np.array(tasklib_test)
            nrow = 2
        fig = plt.figure(figsize=(16, 6 * nrow))
        ax = plt.subplot(nrow, 3, 1)
        plt.plot(trainlib, label="train")
        plt.title("total loss")
        plt.plot(vallib, label="validation")
        plt.plot(testlib, "--", label="test")
        plt.xlabel("Epochs")
        plt.legend()
        plt.yscale("log")

        for ivar in range(tasklib.shape[1]):
            ax = plt.subplot(nrow, 3, 3 + 1 + ivar)
            plt.plot(tasklib[:, ivar], label="train")
            plt.plot(tasklib_vali[:, ivar], label="validation")
            plt.plot(tasklib_test[:, ivar], "--", label="test")
            plt.title("Task " + str(ivar) + ", {:.4f}".format(task_weights[ivar]))
            plt.xlabel("Epochs")
            plt.yscale("log")
            if ivar == 0:
                plt.legend()

        plt.subplots_adjust(
            left=0.1, bottom=0.08, right=0.98, top=0.9, wspace=0.25, hspace=0.3
        )
        fig.savefig(f"./logs/{self.model_with_config_name}/history_loss.png")
        plt.close()
