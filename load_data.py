def load(path, sep='\t'):
    data = []
    with open(path) as fp:
        for line in fp:
            sample = []
            for item in  line.strip().split(sep):
                sample.append(float(item))
            data.append(sample)
    return data