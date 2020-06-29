import matplotlib.pyplot as plt
import re
import pandas as pd
import matplotlib as mpl
import numpy as np
from scipy.fftpack import fftfreq, irfft, rfft
import os
import yaml
import pylab
from scipy import stats
from pyunicorn.timeseries import RecurrencePlot
import pylab
import itertools
import pickle
import json
from statsmodels.tsa.stattools import grangercausalitytests
# from tqdm import tqdm


def find_folder(startpath, folder_name, first_occurrence=False):
    """Find folder by name on specified path.

    :param str startpath: Path on what we will search the folder
    :param str folder_name: Name of folder to search
    :param bool first_occurrence: Whether function should return all occurrences in list format or only the first.

    :return:
        list: List of found paths
   """
    candidates = []
    for root, dirs, files in os.walk(startpath):
        for d in dirs:
            if d == folder_name.strip('/'):
                if first_occurrence:
                    candidates.append(os.path.abspath(root + '/' + d))
                    return candidates
                candidates.append(os.path.abspath(root+'/'+d))
    return candidates


def filter_folders_search(candidates, key):
    """Filter list of folders by key. Here it is the file in the folder.

    :param list[str] candidates: List of paths to filter
    :param str key: Name of file to search in candidate folder
    :return:
        list: List of filtered paths
   """
    output = list()
    for candidate in candidates:
        found = []
        files = os.listdir(candidate)
        if isinstance(key, list):
            for k in key:
                found.append(k in files)
        else:
            found.append(key in files)
        if all(found):
            output.append(candidate)
    return output


def frequency_filter(df, index, dataset_name, experiment_name, datasets_par, analisys_par):
    output = {}
    real_sig = df[index]
    freq = datasets_par[dataset_name]["Experiments"][experiment_name]["sampling rate"]
    duration = datasets_par[dataset_name]["Experiments"][experiment_name]["Duration"]
    dt = 1 / freq
    frequencies = rfft(real_sig)
    W = fftfreq(real_sig.size, d=dt)
    for band in analisys_par["bands"]:
        filtered_frequencies = frequencies.copy()
        filtered_frequencies[(np.abs(W) < analisys_par["bands"][band][0])] = 0
        filtered_frequencies[(np.abs(W) > analisys_par["bands"][band][1])] = 0
        output[band] = irfft(filtered_frequencies)
    return output


def preprocess_data(analysis_par, datasets_par):

    for dataset_name in analysis_par["Datasets"]:
        path = find_folder(proj_path, dataset_name)
        try:
            prep_flag = bool(datasets_par[dataset_name]["Preprocessed"])
        except KeyError:
            prep_flag = False
        if prep_flag:
            continue
        if path:
            datasets_par[dataset_name]["Dataset path"] = path[0]
        else:
            print("Dataset with name: {} not found".format(dataset_name))
            continue
        participants = [os.path.join(datasets_par[dataset_name]["Dataset path"], o)
                        for o in os.listdir(datasets_par[dataset_name]["Dataset path"])
                        if os.path.isdir(os.path.join(datasets_par[dataset_name]["Dataset path"], o))]
        datasets_par[dataset_name]["Dataset participants"] = participants
        ch_f = open(datasets_par[dataset_name]["Dataset path"] + '/' + datasets_par[dataset_name]["Channels_file"], "r")
        ch_n = [re.search(r'\d*\. (.*-.*)', line).group(1) for line in ch_f.readlines()]
        tmp_n = {}
        for i, n in enumerate(ch_n):
            tmp_n[i+1] = n
        for participant in participants:

            # Here we begin to walk through intrested experiments
            for root, dirs, files in os.walk(participant):

                experiment = root.replace(participant, '').strip('/')
                if experiment in analisys_par["Experiments"]:

                    for file in files:
                        intensity = re.search(r'int_(.*)\.dat', file)
                        if not intensity:
                            continue
                        else:
                            intensity = intensity.group(1)
                        df = pd.read_csv(root+'/'+file, sep=r'\s+', header=None)
                        df.rename(tmp_n, axis=1, inplace=True)
                        prep_df = {}
                        for bn in analisys_par["bands"]:
                            prep_df[bn] = pd.DataFrame()
                        for column in df.columns:
                            bands = frequency_filter(df, column, dataset_name,
                                                     experiment, datasets_par, analisys_par)
                            for band in bands:
                                prep_df[band][column] = bands[band]
                        for band in prep_df:
                            save_path = os.path.abspath(participant + '/' + experiment +
                                                        '/' + band + '_int_{}'.format(intensity) + ".csv")
                            prep_df[band].to_csv(save_path)
                        del df, prep_df
        datasets_par[dataset_name]["Preprocessed"] = True
    params_stream = open("analysis.yaml", 'w')
    yaml.safe_dump_all([analisys_par, datasets_par], params_stream)
    params_stream.close()



def load_preprocesed_data(analysis_par, datasets_par, participant_str=None, prep_files=[]):
    dataframes = {}
    for dataset_name in analysis_par["Datasets"]:
        path = find_folder(proj_path, dataset_name)
        if path:
            datasets_par[dataset_name]["Dataset path"] = path[0]
            dataframes[dataset_name] = {}
        else:
            print("Dataset with name: {} not found".format(dataset_name))
            continue
        participants = [os.path.join(datasets_par[dataset_name]["Dataset path"], o)
                        for o in os.listdir(datasets_par[dataset_name]["Dataset path"])
                        if os.path.isdir(os.path.join(datasets_par[dataset_name]["Dataset path"], o))]
        datasets_par[dataset_name]["Dataset participants"] = participants

        for participant in participants:
            dataframes[dataset_name][participant.split('/')[-1]] = {}
            # Here we begin to walk through intrested experiments
            for root, dirs, files in os.walk(participant):
                experiment = root.replace(participant, '').strip('/')
                if experiment in analisys_par["Experiments"]:
                    for file in files:
                        if file in prep_files:
                            prs = re.search(r'(.*)_int_.*\.csv', file)
                            bn = prs.group(1)
                            try:
                                dataframes[dataset_name][participant.split('/')[-1]][bn]
                            except KeyError:
                                dataframes[dataset_name][participant.split('/')[-1]][bn] = {}
                            channel = prs.group(0)
                            dataframes[dataset_name][participant.split('/')[-1]][bn][channel] = \
                                pd.read_csv(participant + '/' + experiment + '/' + channel,
                                            header=0, index_col=0)
                        if not prep_files:
                            for bn in analisys_par["bands"]:
                                channel = re.search(bn + r'_int_(.*)\.csv', file)
                                if channel:
                                    channel = channel.group(0)
                                    try:
                                        dataframes[dataset_name][participant.split('/')[-1]][bn]
                                    except KeyError:
                                        dataframes[dataset_name][participant.split('/')[-1]][bn] = {}
                                    if participant_str == participant.split('/')[-1]:
                                        dataframes[dataset_name][participant.split('/')[-1]][bn][channel] = \
                                        pd.read_csv(participant + '/' + experiment + '/' + channel,
                                                    header=0, index_col=0)
                                    break
    return dataframes


def pearson_coef_inside_band(analisys_par, datasets_par, Participant, plot=False):
    dataframes = load_preprocesed_data(analisys_par, datasets_par, Participant)
    output = {}
    for bn in analisys_par["bands"]:
        output[bn] = {"intencities":[], "pearson":[]}
        for df in dataframes['Proj2Dataset'][Participant][bn]:
            tmp = []
            intensity = re.search(r'int_(.*)\.csv', df)
            output[bn]["intencities"].append(float(intensity.group(1)))
            for i, colx in enumerate(dataframes['Proj2Dataset'][Participant][bn][df].columns):
                for j, coly in enumerate(dataframes['Proj2Dataset'][Participant][bn][df].columns):
                    if i < j and 31 >= i > 0 and 31 >= j > 0:
                        tmp.append(stats.pearsonr(dataframes['Proj2Dataset'][Participant][bn][df][colx],
                                                  dataframes['Proj2Dataset'][Participant][bn][df][coly])[0])
                    else:
                        continue


            output[bn]["pearson"].append(np.mean(tmp))

    for bn in analisys_par["bands"]:
        tmp = sorted(zip(output[bn]["intencities"], output[bn]["pearson"]))
        output[bn]["intencities"], output[bn]["pearson"] = zip(*tmp)

    if plot:
        f = plt.figure()
        for j in analisys_par["bands"]:
            plt.plot(output[j]["intencities"], output[j]["pearson"])
        plt.xlabel('intencities')
        plt.ylabel("pearson coefficient")
        plt.legend(analisys_par["bands"])
        plt.savefig("./Pearson_coeff_inside_band/" + Participant + ".png")
        plt.show()

    return output


def calculate_rp_inside_band(analisys_par, datasets_par, Participant, time_cut=False, plot=False):
    output = {}
    dataframes = load_preprocesed_data(analisys_par, datasets_par, Participant)
    print("Dataframes loaded")

    counter = 0
    for bn in analisys_par["bands"]:
        output[bn] = {"intencities": [], "Mean_RMD": [], "Mean_RMD": {"intencity": "", "channels": np.zeros((31, 31))}}
        for df in dataframes['Proj2Dataset'][Participant][bn]:
            tmp = []
            intensity = re.search(r'int_(.*)\.csv', df)
            output[bn]["intencities"].append(float(intensity.group(1)))
            output[bn]["Mean_RMD"]["intencity"] = intensity.group(1)
            for i, colx in enumerate(dataframes['Proj2Dataset'][Participant][bn][df].columns):
                if 31 >= i > 0:
                    if not time_cut:
                        rpx = RecurrencePlot(
                            np.array(dataframes['Proj2Dataset'][Participant][bn][df][colx]),
                            threshold_std=0.03
                        )
                    else:
                        rpx = RecurrencePlot(
                            np.array(dataframes['Proj2Dataset'][Participant][bn][df][colx][:250 * 15]),
                            threshold_std=0.03
                        )
                    rpx = rpx.recurrence_matrix()
                for j, coly in enumerate(dataframes['Proj2Dataset'][Participant][bn][df].columns):
                    if i < j and 31 >= i > 0 and 31 >= j > 0:
                        counter += 1
                        if not time_cut:
                            rpy = RecurrencePlot(
                                np.array(dataframes['Proj2Dataset'][Participant][bn][df][coly]),
                                threshold_std=0.03
                            )
                        else:
                            rpy = RecurrencePlot(
                                np.array(dataframes['Proj2Dataset'][Participant][bn][df][coly][:250*15]),
                                threshold_std=0.03
                            )
                        rpy = rpy.recurrence_matrix()

                        RMD = np.log2(np.mean((np.mean(np.multiply(rpx, rpy), axis=0) /
                                               (np.mean(rpx, axis=0) * np.mean(rpy, axis=0)))))
                        tmp.append(RMD)
                        output[bn]["Mean_RMD"]["channels"][i-1, j-1] = RMD

                        print("Reccurence_matrix_computed / Total:\n{}/{}".format(counter, 31*31*10))
                    else:
                        continue

            output[bn]["Mean_RMD"].append(np.mean(tmp))

    for bn in analisys_par["bands"]:
        tmp = sorted(zip(output[bn]["intencities"], output[bn]["Mean_RMD"]))
        output[bn]["intencities"], output[bn]["Mean_RMD"] = zip(*tmp)

    if plot:
        f = plt.figure()
        for j in analisys_par["bands"]:
            plt.plot(output[j]["intencities"], output[j]["Mean_RMD"])
        plt.xlabel('intencities')
        plt.ylabel("Mean_RMD")
        plt.legend(analisys_par["bands"])
        plt.savefig("./Mean_RMD_inside_band/" + Participant + ".png")
        plt.show()
    return output

####################################################################################

def coupling_RMD_inside_band(analisys_par, datasets_par, participant, name, dataframe,
                 time_windows=[], time_cut=False, plot=False):
    counter = 0
    output = {"time_window": [], "Mean_RMD": []}
    proj_path = os.path.dirname(os.path.abspath('./project2.py'))
    N = len(time_windows)
    for time_window in time_windows:
        tmp = []
        output["time_window"].append(time_window)
        for i, colx in enumerate(dataframe.columns):
            if 31 >= i > 0:
                if os.path.exists(proj_path + "/tmp/i_{}".format(i) + ".npy"):
                    rpx = np.load(proj_path + "/tmp/i_{}".format(i) + ".npy")
                else:
                    if not time_cut:
                        rpx = np.array(dataframe[colx])
                    else:
                        rpx = np.array(dataframe[colx][:15*250])

                    if time_window > 0:
                        rpx = rpx[:-int(time_window * 250)]

                    if time_window < 0:
                        rpx = rpx[int(time_window * 250):]

                    rpx = RecurrencePlot(rpx, threshold_std=0.03)
                    rpx = rpx.recurrence_matrix()

            for j, coly in enumerate(dataframe.columns):
                if i < j and 31 >= i > 0 and 31 >= j > 0:
                    if os.path.exists(proj_path + "/tmp/j_{}".format(j) + ".npy"):
                        rpy = np.load(proj_path + "/tmp/j_{}".format(j) + ".npy")
                    else:
                        counter += 1
                        if not time_cut:
                            rpy = np.array(dataframe[coly])
                        else:
                            rpy = np.array(dataframe[coly][:15 * 250])

                        if time_window > 0:
                            rpy = rpy[int(time_window * 250):]

                        if time_window < 0:
                            rpy = rpy[:-int(time_window * 250)]

                        rpy = RecurrencePlot(rpy, threshold_std=0.03)
                        rpy = rpy.recurrence_matrix()

                    RMD = np.log2(np.mean((np.mean(np.multiply(rpx, rpy), axis=0) /
                                           (np.mean(rpx, axis=0) * np.mean(rpy, axis=0)))))
                    tmp.append(RMD)
                    print("Reccurence_matrix_computed / Total:\n{}/{}".format(counter, (31*31/2-31)*N))
                else:
                    continue

        output["Mean_RMD"].append(np.mean(tmp))

    tmp = sorted(zip(output["time_window"], output["Mean_RMD"]))
    output["time_window"], output["Mean_RMD"] = zip(*tmp)

    if plot:
        f = plt.figure()
        plt.plot(output["time_window"], output["Mean_RMD"])
        plt.xlabel('time_window')
        plt.ylabel("Mean_RMD")
        bn = re.search(r'(.*)_int_.*\.csv', name)
        plt.legend(bn.group(1))
        plt.savefig("./Coupling_strength_inside_band/" + participant + '_' + name + ".png")
        plt.show()
    return output

####################################################################################


def calculate_rp_between_bands(analisys_par, datasets_par, Participant, time_cut=False, plot=False):
    output = {}
    dataframes = load_preprocesed_data(analisys_par, datasets_par, Participant)
    print("Dataframes loaded")
    proj_path = os.path.dirname(os.path.abspath('./project2.py'))

    counter = 0
    comb_list = itertools.combinations(list(analisys_par["bands"]), 2)

    for bands in comb_list:
        output[bands[0]+'-'+bands[1]] = {"intencities": [], "Mean_RMD": [], "Mean_diag_RMD": [],
                      "Mean_RMD_mat": {"intencity": [], "channels": []}}
        dt_frames_1 = dataframes['Proj2Dataset'][Participant][bands[0]]
        dt_frames_2 = dataframes['Proj2Dataset'][Participant][bands[1]]
        for k, intensity in enumerate(range(1, 11)):
            intensity = round(intensity * 0.1, 1)
            if intensity == 1.0:
                intensity = int(intensity)
            df1 = dt_frames_1[bands[0] + "_int_{}.csv".format(intensity)]
            df2 = dt_frames_2[bands[1] + "_int_{}.csv".format(intensity)]
            tmp_overhaul = []
            tmp_diag = []
            output[bands[0]+'-'+bands[1]]["intencities"].append(intensity)
            output[bands[0]+'-'+bands[1]]["Mean_RMD_mat"]["intencity"].append(intensity)
            output[bands[0]+'-'+bands[1]]["Mean_RMD_mat"]["channels"].append(np.zeros((31, 31)))

            for i, colx in enumerate(df1.columns):
                if 31 >= i > 0 :
                    if os.path.exists(proj_path + "/tmp/i_{}".format(i) + ".npy"):
                        rpx = np.load(proj_path + "/tmp/i_{}".format(i) + ".npy")
                    else:
                        if not time_cut:
                            rpx = RecurrencePlot(
                                np.array(df1[colx]),
                                threshold_std=0.03
                            )
                        else:
                            rpx = RecurrencePlot(
                                np.array(df1[colx][:250 * 15]),
                                threshold_std=0.03
                            )
                        rpx = rpx.recurrence_matrix()
                        np.save(proj_path + "/tmp/i_{}".format(i), rpx)

                for j, coly in enumerate(df2.columns):
                    if 31 >= i > 0 and 31 >= j > 0:
                        counter += 1
                        if os.path.exists(proj_path + "/tmp/j_{}".format(j) + ".npy"):
                            rpy = np.load(proj_path + "/tmp/j_{}".format(j) + ".npy")
                        else:
                            if not time_cut:
                                rpy = RecurrencePlot(
                                    np.array(df2[coly]),
                                    threshold_std=0.03
                                )
                            else:
                                rpy = RecurrencePlot(
                                    np.array(df2[coly][:250*15]),
                                    threshold_std=0.03
                                )
                            rpy = rpy.recurrence_matrix()
                            np.save(proj_path + "/tmp/j_{}".format(j), rpy)

                        RMD = np.log2(np.mean((np.mean(np.multiply(rpx, rpy), axis=0) /
                                               (np.mean(rpx, axis=0) * np.mean(rpy, axis=0)))))
                        tmp_overhaul.append(RMD)
                        if i == j:
                            tmp_diag.append(RMD)
                        output[bands[0]+'-'+bands[1]]["Mean_RMD_mat"]["channels"][k][i-1, j-1] = RMD

                        print("Reccurence_matrix_computed / Total:\n{}/{}".format(counter, 31*31*10))
                    else:
                        continue

            output[bands[0]+'-'+bands[1]]["Mean_RMD"].append(np.mean(tmp_overhaul))

            output[bands[0] + '-' + bands[1]]["Mean_diag_RMD"].append(np.mean(tmp_diag))

            for i, colx in enumerate(df1.columns):
                if os.path.exists(proj_path + "/tmp/i_{}".format(i) + ".npy"):
                    os.remove(proj_path + "/tmp/i_{}".format(i) + ".npy")
            for j, coly in enumerate(df2.columns):
                if os.path.exists(proj_path + "/tmp/j_{}".format(j) + ".npy"):
                    os.remove(proj_path + "/tmp/j_{}".format(j) + ".npy")

    # for bands in comb_list:
    #     tmp = sorted(zip(output[bands[0]+'-'+bands[1]]["intencities"],
    #                      output[bands[0]+'-'+bands[1]]["Mean_RMD"]))
    #     output[bands[0]+'-'+bands[1]]["intencities"], output[bands[0]+'-'+bands[1]]["Mean_RMD"] = zip(*tmp)

    with open("./Mean_RMD_between_bands/" + Participant + '.pickle', 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if plot:
        comb_list = itertools.combinations(list(analisys_par["bands"]), 2)
        fig1 = plt.figure()
        cumul_lgn = []
        for bands in comb_list:
            pair = bands[0]+'-'+bands[1]
            plt.plot(output[pair]["intencities"][:], output[pair]["Mean_RMD"][:])
            cumul_lgn.append(pair)
        plt.xlabel('intencities')
        plt.ylabel("Mean_RMD")
        plt.legend(cumul_lgn)
        plt.savefig("./Mean_RMD_between_bands/" + Participant + ".png")
        plt.show()


        comb_list = itertools.combinations(list(analisys_par["bands"]), 2)
        fig2 = plt.figure()
        cumul_lgn = []
        for bands in comb_list:
            pair = bands[0] + '-' + bands[1]
            plt.plot(output[pair]["intencities"][:], output[pair]["Mean_diag_RMD"][:])
            cumul_lgn.append(pair)
        plt.xlabel('intencities')
        plt.ylabel("Mean_diag_RMD")
        plt.legend(cumul_lgn)
        plt.savefig("./Mean_RMD_between_bands/" + Participant + "_diag.png")
        plt.show()

    return output

####################################################################################

def coupling_RMD_between_bands(analisys_par, datasets_par, participant, intencity,
                             time_windows=[], time_cut=False, plot=False):
    counter = 0
    output = {}
    proj_path = os.path.dirname(os.path.abspath('./project2.py'))
    dataframes = load_preprocesed_data(analisys_par, datasets_par, participant)
    comb_list = itertools.combinations(list(analisys_par["bands"]), 2)
    N = len(time_windows)

    for bands in comb_list:
        output[bands[0]+'-'+bands[1]] = {"time_window": [], "Mean_RMD": []}
        for time_window in time_windows:
            tmp = []
            output[bands[0]+'-'+bands[1]]["time_window"].append(time_window)
            dt_frames_1 = dataframes['Proj2Dataset'][Participant][bands[0]]
            dt_frames_2 = dataframes['Proj2Dataset'][Participant][bands[1]]
            if intencity == 1.0:
                intencity = int(intencity)
            df1 = dt_frames_1[bands[0] + "_int_{}.csv".format(intencity)]
            df2 = dt_frames_2[bands[1] + "_int_{}.csv".format(intencity)]
            for i, colx in enumerate(df1.columns):
                if 31 >= i > 0:
                    if os.path.exists(proj_path + "/tmp/i_{}".format(i) + ".npy"):
                        rpx = np.load(proj_path + "/tmp/i_{}".format(i) + ".npy")
                    else:
                        if not time_cut:
                            rpx = np.array(df1[colx])
                        else:
                            rpx = np.array(df1[colx][:15 * 250])

                        if time_window > 0:
                            rpx = rpx[:-int(time_window * 250)]

                        if time_window < 0:
                            rpx = rpx[int(time_window * 250):]

                        rpx = RecurrencePlot(rpx, threshold_std=0.03)
                        rpx = rpx.recurrence_matrix()
                        np.save(proj_path + "/tmp/i_{}".format(i), rpx)

                for j, coly in enumerate(df2.columns):
                    if 31 >= i > 0 and 31 >= j > 0:
                        counter += 1
                        if os.path.exists(proj_path + "/tmp/j_{}".format(j) + ".npy"):
                            rpy = np.load(proj_path + "/tmp/j_{}".format(j) + ".npy")
                        else:
                            if not time_cut:
                                rpy = np.array(df2[coly])
                            else:
                                rpy = np.array(df2[coly][:15 * 250])

                            if time_window > 0:
                                rpy = rpy[int(time_window * 250):]

                            if time_window < 0:
                                rpy = rpy[:-int(time_window * 250)]

                            rpy = RecurrencePlot(rpy, threshold_std=0.03)
                            rpy = rpy.recurrence_matrix()
                            np.save(proj_path + "/tmp/j_{}".format(j), rpy)

                        RMD = np.log2(np.mean((np.mean(np.multiply(rpx, rpy), axis=0) /
                                               (np.mean(rpx, axis=0) * np.mean(rpy, axis=0)))))
                        tmp.append(RMD)
                        print("Reccurence_matrix_computed / Total:\n{}/{}".format(counter, (31 * 31) * N))
                    else:
                        continue

            output[bands[0]+'-'+bands[1]]["Mean_RMD"].append(np.mean(tmp))

            for i, colx in enumerate(df1.columns):
                if os.path.exists(proj_path + "/tmp/i_{}".format(i) + ".npy"):
                    os.remove(proj_path + "/tmp/i_{}".format(i) + ".npy")
            for j, coly in enumerate(df2.columns):
                if os.path.exists(proj_path + "/tmp/j_{}".format(j) + ".npy"):
                    os.remove(proj_path + "/tmp/j_{}".format(j) + ".npy")

    if plot:
        f = plt.figure()
        lgn = []
        comb_list = itertools.combinations(list(analisys_par["bands"]), 2)
        for bands in comb_list:
            plt.plot(output[bands[0]+'-'+bands[1]]["time_window"],
                     output[bands[0]+'-'+bands[1]]["Mean_RMD"])
            lgn.append(bands[0]+'-'+bands[1])
        plt.xlabel('time_window')
        plt.ylabel("Mean_RMD")
        plt.legend(lgn)
        plt.savefig("./Coupling_strength_between_bands/" + participant +
                    '_int_' + str(intencity) + ".png")
        plt.show()
    return output

####################################################################################


def Non_parametric_test(analisys_par, datasets_par, participant, intencity, band1, band2,
                             time_windows=[], time_cut=False, plot=False):
    counter = 0
    output = {}
    proj_path = os.path.dirname(os.path.abspath('./project2.py'))
    dataframes = load_preprocesed_data(analisys_par, datasets_par, participant)
    comb_list = itertools.combinations(list(analisys_par["bands"]), 2)
    N = len(time_windows)

    output = {"time_window": [], "Mean_RMD": []}
    for time_window in time_windows:
        tmp = []
        output[bands[0]+'-'+bands[1]]["time_window"].append(time_window)
        dt_frames_1 = dataframes['Proj2Dataset'][Participant][bands[0]]
        dt_frames_2 = dataframes['Proj2Dataset'][Participant][bands[1]]
        if intencity == 1.0:
            intencity = int(intencity)
        df1 = dt_frames_1[bands[0] + "_int_{}.csv".format(intencity)]
        df2 = dt_frames_2[bands[1] + "_int_{}.csv".format(intencity)]
        for i, colx in enumerate(df1.columns):
            if 31 >= i > 0:
                if os.path.exists(proj_path + "/tmp/i_{}".format(i) + ".npy"):
                    rpx = np.load(proj_path + "/tmp/i_{}".format(i) + ".npy")
                else:
                    if not time_cut:
                        rpx = np.array(df1[colx])
                    else:
                        rpx = np.array(df1[colx][:15 * 250])

                    if time_window > 0:
                        rpx = rpx[:-int(time_window * 250)]

                    if time_window < 0:
                        rpx = rpx[int(time_window * 250):]

                    rpx = RecurrencePlot(rpx, threshold_std=0.03)
                    rpx = rpx.recurrence_matrix()
                    np.save(proj_path + "/tmp/i_{}".format(i), rpx)

            for j, coly in enumerate(df2.columns):
                if 31 >= i > 0 and 31 >= j > 0:
                    counter += 1
                    if os.path.exists(proj_path + "/tmp/j_{}".format(j) + ".npy"):
                        rpy = np.load(proj_path + "/tmp/j_{}".format(j) + ".npy")
                    else:
                        if not time_cut:
                            rpy = np.array(df2[coly])
                        else:
                            rpy = np.array(df2[coly][:15 * 250])

                        if time_window > 0:
                            rpy = rpy[int(time_window * 250):]

                        if time_window < 0:
                            rpy = rpy[:-int(time_window * 250)]

                        rpy = RecurrencePlot(rpy, threshold_std=0.03)
                        rpy = rpy.recurrence_matrix()
                        np.save(proj_path + "/tmp/j_{}".format(j), rpy)

                    RMD = np.log2(np.mean((np.mean(np.multiply(rpx, rpy), axis=0) /
                                           (np.mean(rpx, axis=0) * np.mean(rpy, axis=0)))))
                    tmp.append(RMD)
                    print("Reccurence_matrix_computed / Total:\n{}/{}".format(counter, (31 * 31) * N))
                else:
                    continue

        output[bands[0]+'-'+bands[1]]["Mean_RMD"].append(np.mean(tmp))

        for i, colx in enumerate(df1.columns):
            if os.path.exists(proj_path + "/tmp/i_{}".format(i) + ".npy"):
                os.remove(proj_path + "/tmp/i_{}".format(i) + ".npy")
        for j, coly in enumerate(df2.columns):
            if os.path.exists(proj_path + "/tmp/j_{}".format(j) + ".npy"):
                os.remove(proj_path + "/tmp/j_{}".format(j) + ".npy")

    if plot:
        f = plt.figure()
        lgn = []
        comb_list = itertools.combinations(list(analisys_par["bands"]), 2)
        for bands in comb_list:
            plt.plot(output[bands[0]+'-'+bands[1]]["time_window"],
                     output[bands[0]+'-'+bands[1]]["Mean_RMD"])
            lgn.append(bands[0]+'-'+bands[1])
        plt.xlabel('time_window')
        plt.ylabel("Mean_RMD")
        plt.legend(lgn)
        plt.savefig("./Coupling_strength_between_bands/" + participant +
                    '_int_' + str(intencity) + ".png")
        plt.show()
    return output


####################################################################################


proj_path = os.path.dirname(os.path.abspath('./project2.py'))
params_stream = open("analysis.yaml", 'r')
analisys_par, datasets_par = yaml.safe_load_all(params_stream)
params_stream.close()

# preprocess_data(analisys_par, datasets_par)

Participant = "Participant 5"
# dataframes = load_preprocesed_data(analisys_par, datasets_par, Participant)

# pearson_coef(analisys_par, datasets_par, Participant, plot=True)

# personDict = calculate_rp_inside_band(analisys_par, datasets_par, Participant, time_cut=True, plot=True)
# personDict = calculate_rp_between_bands(analisys_par, datasets_par, Participant, time_cut=True, plot=True)

# coupling_RMD_inside_band()
# coupling_RMD_between_bands()

# targets = {"Participant 1": ["beta_int_0.7.csv", "alpha_int_0.9.csv"],
#            "Participant 2": ["beta_int_0.6.csv", "alpha_int_0.5.csv"],
#            "Participant 3": ["beta_int_0.7.csv", "alpha_int_0.4.csv"],
#            "Participant 4": ["beta_int_0.2.csv", "alpha_int_0.3.csv"],
#            "Participant 5": ["beta_int_0.5.csv", "alpha_int_0.3.csv"]}

time_windows = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# for participant in targets:
#     prep_files = targets[participant]
#     target_dataframes = \
#         load_preprocesed_data(analisys_par, datasets_par, participant, prep_files)
#     for bn in analisys_par["bands"]:
#         for df in target_dataframes['Proj2Dataset'][participant][bn]:
#             coupling_RMD_inside_band(analisys_par, datasets_par, participant, df,
#                          target_dataframes['Proj2Dataset'][participant][bn][df],
#                          time_windows=time_windows, time_cut=True, plot=True)

# coupling_RMD_between_bands(analisys_par, datasets_par, Participant, 0.7, time_windows,
#                            True, True)










