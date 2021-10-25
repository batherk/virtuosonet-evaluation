import torch
from utils.conversion import convert_latent_to_model_z, convert_model_z_to_latent

class Style:

    def __init__(self, qpm_primo, latent, name=''):
        self.qpm_primo = qpm_primo
        self.name = name

        if type(latent[0]) == torch.Tensor:
            self.latent = convert_model_z_to_latent(latent)
        else:
            self.latent = latent

    def to_dict(self):
        return {
            'z': convert_latent_to_model_z(self.latent),
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
    latent=[ -0.8611,  0.1509,  0.1954, -0.1998, -0.0693, -0.3300, -0.1151,
           0.4470, -0.0036,  0.2544,  0.1300,  0.3184,  0.3081, -0.0575,
           0.1970,  0.0145]
)

sad = Style(
    name='Sad',
    qpm_primo=-1.3659845383895572,
    latent=[0.3559,  0.0670, -0.0238,  0.4218, -0.0883, -0.1958, -0.0273,
           0.1367, -0.0793,  0.0682,  0.1752,  0.2429,  0.0364, -0.3415,
           0.2484, -0.1466]
)

relax = Style(
    name='Relax',
    qpm_primo=-0.7279451943405377,
    latent=[-0.1730, -0.1018, -0.0196,  0.4134, -0.3197,  0.2387, -0.3057,
          -0.2355,  0.2783, -0.1661, -0.6581,  0.4719,  0.0260, -0.3694,
           0.5658,  0.3900]
)


