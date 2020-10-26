from docopt import docopt


def generate_results_color():
    pass


def get_bbdd(folder):
    pass


def get_dataset(param):
    pass


def background_removal_test():
    pass


def color_noise(dataset, method):
    bbdd = get_bbdd('../BBDD')
    dataset = get_dataset(dataset)

    # Compute descriptors
    bbdd_descriptors = []
    dataset_descriptors = []

    # Generate results
    generate_results_color(bbdd_descriptors, dataset_descriptors)


if __name__ == '__main__':
    args = docopt(__doc__)

    week = int(args['<weekNumber>'])  # 1
    team = int(args['<teamNumber>'])  # 04
    query_set = int(args['<querySet>'])  # 1 or 2
    method = int(args['<MethodNumber>'])  # 1: divided_hist  2:rgb_3d
    distance_m = int(args['<distanceMeasure>'])  # 1: euclidean and 2: x^2 distance

    dataset = 'qsd{}_w{}'.format(query_set, week)

    # Call to the test
    color_noise(dataset, method)
