import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    syntesized_spec_path = "/mnt/hdd1/adibian/FastSpeech2/Single-Speaker-FastSpeech2-new-spec/output/synthesized_specs/"
    real_spec_path = "/mnt/hdd1/adibian/FastSpeech2/Single-Speaker-FastSpeech2-new-spec/preprocessed_data/Persian/mel/"
    info1 = {}
    info2 = {}
    count = 0
    for d1 in os.listdir(syntesized_spec_path):
        print(d1)
        # count += 1
        # if count>10:
        #     break
        sample_file = os.path.join(syntesized_spec_path, d1)
        if os.path.isfile(sample_file):
            a = np.load(sample_file)
            info1[d1] = a.shape[2]

    count = 0
    for d1 in os.listdir(real_spec_path):
    # for d1 in info1.keys():
        # count += 1
        # if count>10:
        #     break
        print(d1)
        # sample_file = os.path.join(real_spec_path, 'single_speaker-mel-'+d1)
        sample_file = os.path.join(real_spec_path, d1)
        if os.path.isfile(sample_file):
            a = np.load(sample_file)
            info2[d1.replace('single_speaker-mel-', '')] = a.shape[0]

    diff_shape = []
    print(info1)
    print(info2)
    for file_name in info1.keys():
        diff_shape.append(info1[file_name] - info2[file_name])

    plt.hist(diff_shape)
    plt.title("difference of the number of frames in real and predicted spec")
    plt.savefig("output.jpg", bins=35)
    plt.clf()

    for file_name in info1.keys():
        diff = abs(info1[file_name] - info2[file_name])
        if diff > 0:
            vec = np.load(syntesized_spec_path + file_name).squeeze()
            print(vec.shape)
            plt.imshow(vec, origin ='lower')
            title = file_name[:file_name.find('.')] + '_predicted_' + str(diff)
            plt.title(title)
            plt.savefig(os.path.join('specs', title + '.jpg'))
            plt.clf()
            vec = np.load(real_spec_path + 'single_speaker-mel-' + file_name).squeeze()
            print(vec.T.shape)
            plt.imshow(vec.T, origin ='lower')
            title = file_name[:file_name.find('.')] + '_real_' + str(diff)
            plt.title(title)
            plt.savefig(os.path.join('specs', title + '.jpg'))
            plt.clf()


