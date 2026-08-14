import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

import ptychodus.api.simulate._phase_unwrap as pu


@pytest.mark.skip(reason='this test relies on data that has been excluded from the git repo.')
def test_phase_unwrap() -> None:
    phase_unwrapper = pu.PhaseUnwrapper()
    img = np.load(os.path.join('data', 'phase_unwrap', 'recon_20241220_epoch_400.npy'))
    img = img[0]

    phase = phase_unwrapper.unwrap(img)

    plt.figure()
    plt.imshow(phase)
    plt.show()


if __name__ == '__main__':
    test_phase_unwrap()
