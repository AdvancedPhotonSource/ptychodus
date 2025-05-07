import os

import numpy as np
import matplotlib.pyplot as plt

import ptychodus.model.phase_unwrapper as pu


def test_phase_unwrap() -> None:
    phase_unwrapper = pu.PhaseUnwrapper(
        image_grad_method='fourier_differentiation',
        image_integration_method='fourier',
    )
    img = np.load(os.path.join('data', 'phase_unwrap', 'recon_20241220_epoch_400.npy'))
    img = img[0]

    phase = phase_unwrapper.unwrap(img)

    plt.figure()
    plt.imshow(phase)
    plt.show()


if __name__ == '__main__':
    test_phase_unwrap()
