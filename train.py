import torch
import json
import optparse


def train_model(train, test, loss=None, optimizer=None, device='cpu'):
    ''' train_model(train, test, ...) -> (HeroVillainNet, stats)

    Trains a HeroVillainNet model, given the training and test data in
    'train' and 'test', respectively. Returns a trained model and a dict
    containing accuracy statistics for the trained model.
    '''
    raise NotImplementedError

def save_model(model, filename):
    state = model.state_dict()
    serialized_state = json.dumps(state)
    with open(filename, "w") as f:
        f.write(serialized_state)


if __name__ == "__main__":
    p = optparse.OptionParser(
        usage="Usage: %prog [OPTIONS] IMAGE_DIR IMG_LABEL_MAP",
        description="Trains a convolutional neural network based on the "
        "images contained in IMAGE_DIR, labeled by the data in IMG_LABEL_MAP."
    )
    p.add_option('-d', '--device', dest='device', default='cpu',
                 help='run pytorch operations on a specific device')
    opts, args = p.parse_args()

    raise NotImplementedError
