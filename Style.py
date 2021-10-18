import torch

class Style:

    def __init__(self, qpm_primo, latent, name=''):
        self.qpm_primo = qpm_primo
        self.latent = latent
        self.name=name

    def to_dict(self):
        return {
            'z': [torch.tensor([[self.latent]])],
            'key': self.name,
            'qpm': self.qpm_primo
        }

    def __add__(self, other):
        qpm_primo = self.qpm_primo + other.qpm_primo
        latent = [a + b for a, b in zip(self.latent, other.latent)]
        return Style(qpm_primo, latent)

    def __sub__(self, other):
        qpm_primo = self.qpm_primo - other.qpm_primo
        latent = [a - b for a, b in zip(self.latent, other.latent)]
        return Style(qpm_primo, latent)

    def approach(self, other, percent, name=''):
        diff = other.__sub__(self)
        qpm_primo = self.qpm_primo + diff.qpm_primo * percent
        latent = [a + b * percent for a, b in zip(self.latent, diff.latent)]
        return Style(qpm_primo, latent, name)

anger = Style(
    name='Anger',
    qpm_primo=-0.0881854666908756,
    latent=[ 0.5749, -0.1427, -0.1627,  0.1102, -0.0582,  0.5274, -0.1459,
           0.0134, -0.1858,  0.2948, -0.0082,  0.0314,  0.2051,  0.3028,
           0.0732,  0.0530]
)

sad = Style(
    name='Sad',
    qpm_primo=-1.3659845383895572,
    latent=[-0.0133, -0.0521, -0.1571, -0.0232,  0.0296, -0.2825,  0.1783,
           0.7856, -0.0367, -0.6838, -0.0488, -0.2774, -0.1716, -0.0218,
           0.2626,  0.8341]
)

relax = Style(
    name='Relax',
    qpm_primo=-0.7279451943405377,
    latent=[-0.3656, -0.0048, -0.2384,  0.0565, -0.1240, -0.1579,  0.0561,
          -0.5261, -0.0658,  0.4233, -0.0478, -0.0314, -0.5431, -0.2640,
           0.3206, -0.1124]
)


