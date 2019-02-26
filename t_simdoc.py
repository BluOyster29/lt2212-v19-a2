import argparse, pandas, numpy
from sklearn.metrics.pairwise import cosine_similarity     #this did not work :(

parser = argparse.ArgumentParser()
parser.add_argument('f', metavar='filename', help='takes the name of the file to be opened')
args = parser.parse_args()


def separate_datafile(file):
    """
    Reads a csv file and separates it into distinct topics
    :param file: csv file
    :return array: two nparray objects associated with each topic
    """

    data_matrix = pandas.read_csv(file, index_col=False).values

    grain_array = []
    crude_array = []
    for value in data_matrix:
        if 'grain' in value[0]:
            grain_array.append(value[1:3])
        else:
            crude_array.append(value[1:3])

    return grain_array, crude_array


def cos_similarity():
    """
    calculates the average cosine similarity of the document vectors within each topic, as well as between them
    """

    cos_sim_grain = []
    cos_sim_crude = []
    cos_sim_grainXcrude = []
    cos_sim_crudeXgrain = []

    for i in range(len(grain)):
        dot = numpy.dot(grain[i], grain[::-1][i])
        norm_ga = numpy.linalg.norm(grain[i])
        norm_gz = numpy.linalg.norm(grain[::-1][i])
        if norm_ga and norm_gz != 0:
            cos_sim_grain.append(dot/(norm_ga*norm_gz))
    average_cos_grain = sum(cos_sim_grain)/len(cos_sim_grain)
    print("Average similarity of document vectors within the grain topic:", average_cos_grain)

    for i in range(len(crude)):
        dot = numpy.dot(crude[i], crude[::-1][i])
        norm_ca = numpy.linalg.norm(crude[i])
        norm_cz = numpy.linalg.norm(crude[::-1][i])
        if norm_ca and norm_cz != 0:
            cos_sim_crude.append(dot/(norm_ca*norm_cz))
    average_cos_crude = sum(cos_sim_crude)/len(cos_sim_crude)
    print("Average similarity of document vectors within the crude topic:", average_cos_crude)

    for i in range(len(grain)):
        for j in range(len(crude)):
            dot = numpy.dot(grain[i], crude[j])
            norm_g = numpy.linalg.norm(grain[i])
            norm_c = numpy.linalg.norm(crude[j])
            if norm_g and norm_c != 0:
                cos_sim_grainXcrude.append(dot/(norm_g*norm_c))
    average_cos_grainXcrude = sum(cos_sim_grainXcrude)/len(cos_sim_grainXcrude)
    print("Average similarity of document vectors in grain to the document vectors in crude:", average_cos_grainXcrude)

    for y in range(len(crude)):
        for z in range(len(grain)):
            dot = numpy.dot(crude[y], grain[z])
            norm_g = numpy.linalg.norm(grain[z])
            norm_c = numpy.linalg.norm(crude[y])
            if norm_g and norm_c != 0:
                cos_sim_crudeXgrain.append(dot/(norm_g*norm_c))
    average_cos_crudeXgrain = sum(cos_sim_crudeXgrain)/len(cos_sim_crudeXgrain)
    print("Average similarity of document vectors in crude to the document vectors in grain:", average_cos_crudeXgrain)


if __name__ == '__main__':
    grain, crude = separate_datafile(args.f)
    cos_similarity()