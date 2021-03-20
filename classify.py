import torch
import hvmodel
import json
import optparse

def classify_character(model, name, image):
    ''' classify_character(str, image) -> "Hero"|"Villain"

    Uses the trained HeroVillainNet 'model' to predict whether the given
    character image/name is a hero or a villain, returning the string
    "hero" or "villain" according to the model's prediction.
    '''
    raise NotImplementedError

def load_model(state_filename, device='cpu'):
    ''' load_model(state_filename) -> HeroVillainNet

    Loads the state data of a trained model from 'state_filename', and
    returns an instance of HeroVillainNet with the training data
    loaded.
    '''

    model = hvmodel.HeroVillainNet(device=device)

    with open(state_filename, "r") as f:
        serialized_state = f.read()

    state = json.loads(serialized_state)
    model.load_state_dict(state)

    return model


if __name__ == "__main__":
    p = optparse.OptionParser(
        usage="Usage: %prog [OPTIONS] CNN_STATE IMAGE",
        description="Classifies an image as 'hero' or 'villain' based on the "
        "output of the trained convolutional neural network."
    )
    p.add_option('-d', '--device', dest='device', default='cpu',
                 help='run pytorch operations on a specific device')
    opts, args = p.parse_args()

    raise NotImplementedError
