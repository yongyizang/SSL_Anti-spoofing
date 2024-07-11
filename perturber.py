import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import soundfile as sf

class Perturber(nn.Module):
    def __init__(self, window_size, hop_size, perturb_setting, device):
        super(Perturber, self).__init__()
        self.window_size = window_size
        self.hop_size = hop_size
        self.perturb_setting = perturb_setting
        self.device = torch.device(device)
        
        # Magnitude
        self.perturb_magnitude_per_row = perturb_setting['perturb_magnitude_per_row']
        self.perturb_magnitude_per_column = perturb_setting['perturb_magnitude_per_column']
        self.perturb_magnitude = False
        if self.perturb_magnitude_per_row or self.perturb_magnitude_per_column:
            self.perturb_magnitude = True
            
        # Phase
        self.perturb_phase_per_row = perturb_setting['perturb_phase_per_row']
        self.perturb_phase_per_column = perturb_setting['perturb_phase_per_column']
        self.perturb_phase = False
        if self.perturb_phase_per_row or self.perturb_phase_per_column:
            self.perturb_phase = True
            
        self.magnitude_perturb_amount = perturb_setting['magnitude_perturb_amount'] # in SNR
        self.phase_perturb_amount = perturb_setting['phase_perturb_amount'] # between 0 to 1, 0 means no perturb, 1 means random phase (perturbation=2*pi)
        
        assert self.phase_perturb_amount >= 0 and self.phase_perturb_amount <= 1, "Phase perturbation amount should be between 0 and 1"

    def forward(self, audio_tensor):        
        if self.perturb_magnitude:
            target_SNR = 10 ** (self.magnitude_perturb_amount / 10)  # Convert SNR from dB to ratio
            power = torch.mean(audio_tensor ** 2)
            noise = torch.randn_like(audio_tensor, device=self.device)
            noise_power = torch.mean(noise ** 2)

            scaling_factor = torch.sqrt(power / (noise_power * target_SNR))
            noise = noise * scaling_factor
            noisy_audio = audio_tensor + noise
            
            noisy_stft = torch.stft(
                noisy_audio,
                n_fft=self.window_size,
                hop_length=self.hop_size,
                window=torch.hann_window(self.window_size).to(self.device),
                return_complex=True,
            )

        stft = torch.stft(
            audio_tensor,
            n_fft=self.window_size,
            hop_length=self.hop_size,
            window=torch.hann_window(self.window_size).to(self.device),
            return_complex=True,
        )
        
        magnitude = torch.abs(stft)
        
        """
        If the entire magnitude spectrogram is perturbed,
        we directly use waveform-level SNR.
        Otherwise, we calculate the perturbation mask and apply it to the magnitude spectrogram.
        The SNR is then calculated based on the perturbation mask vs the original magnitude spectrogram.
        """
        
        if self.perturb_magnitude:
            magnitude = torch.abs(noisy_stft)
            
        
        elif self.perturb_magnitude_per_row:
            row_shape = magnitude.shape[0]
            perturb_mask = torch.rand(row_shape, device=self.device) * 2 * self.magnitude_perturb_amount - self.magnitude_perturb_amount
            perturb_mask = torch.tile(perturb_mask[:, None], (1, magnitude.shape[1]))
            target_Snr = 10 ** (self.magnitude_perturb_amount / 10)
            power = magnitude ** 2
            noise_power = power / target_Snr
            noise_power = noise_power + 1e-10
            scaling_factor = torch.sqrt(power / noise_power)
            magnitude = magnitude + perturb_mask * scaling_factor
            
        elif self.perturb_magnitude_per_column:
            column_shape = magnitude.shape[1]
            perturb_mask = torch.rand(column_shape, device=self.device) * 2 * self.magnitude_perturb_amount - self.magnitude_perturb_amount
            perturb_mask = torch.tile(perturb_mask[None, :], (magnitude.shape[0], 1))
            target_Snr = 10 ** (self.magnitude_perturb_amount / 10)
            power = magnitude ** 2
            noise_power = power / target_Snr
            noise_power = noise_power + 1e-10
            scaling_factor = torch.sqrt(power / noise_power)
            magnitude = magnitude + perturb_mask * scaling_factor
            
        phase = torch.angle(stft)
        
        """
        For phase, we perturb the phase spectrogram directly.
        """

        if self.perturb_phase:
            perturb_noise = torch.rand(phase.shape, device=self.device) * 2 * np.pi
            perturb_noise = perturb_noise - np.pi
            perturb_noise = perturb_noise * self.phase_perturb_amount
            phase = phase + perturb_noise
            
        elif self.perturb_phase_per_row:
            row_shape = phase.shape[0]
            perturb_mask = torch.rand(row_shape, device=self.device) * 2 * np.pi
            perturb_mask = perturb_mask - np.pi
            perturb_mask = perturb_mask * self.phase_perturb_amount
            perturb_noise = torch.tile(perturb_mask[:, None], (1, phase.shape[1]))
            phase = phase + perturb_noise
            
        elif self.perturb_phase_per_column:
            column_shape = phase.shape[1]
            perturb_mask = torch.rand(column_shape, device=self.device) * 2 * np.pi
            perturb_mask = perturb_mask - np.pi
            perturb_mask = perturb_mask * self.phase_perturb_amount
            perturb_noise = torch.tile(perturb_mask[None, :], (phase.shape[0], 1))
            phase = phase + perturb_noise
            
        perturbed_stft = magnitude * torch.exp(1j * phase)

        # Inverse STFT to obtain the perturbed audio
        perturbed_audio = torch.istft(
            perturbed_stft,
            n_fft=self.window_size,
            hop_length=self.hop_size,
            window=torch.hann_window(self.window_size).to(self.device),
        )

        return perturbed_audio

import torch
import torch.nn as nn
import math

class PhasePerturber(nn.Module):
    def __init__(self, window_size, hop_size, perturb_setting, device):
        super(PhasePerturber, self).__init__()
        self.window_size = window_size
        self.hop_size = hop_size
        self.perturb_setting = perturb_setting
        self.device = torch.device(device)
        
        # Phase
        self.perturb_phase_per_row = perturb_setting['perturb_phase_per_row']
        self.perturb_phase_per_column = perturb_setting['perturb_phase_per_column']
        # self.perturb_phase = False
        # if self.perturb_phase_per_row or self.perturb_phase_per_column:
        #     self.perturb_phase = True
            
        self.phase_perturb_dist = torch.distributions.Normal(0, 1)

    def forward(self, audio_tensor):
        original_length = audio_tensor.shape[-1]

        stft = torch.stft(
            audio_tensor,
            n_fft=self.window_size,
            hop_length=self.hop_size,
            window=torch.hann_window(self.window_size).to(self.device),
            return_complex=True
        )
        
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        phase_perturb_amount = self.phase_perturb_dist.sample(torch.Size([phase.shape[0]])).to(self.device)
        
        if self.perturb_phase_per_row and self.perturb_phase_per_column:
            perturb_noise = torch.randn(phase.shape, device=self.device)
            perturb_noise = perturb_noise * phase_perturb_amount[:, None, None]
            phase = phase + perturb_noise
            
        elif self.perturb_phase_per_row:
            row_shape = torch.Size([phase.shape[0], phase.shape[1]])
            perturb_mask = torch.randn(row_shape, device=self.device)
            phase_perturb_amount = torch.tile(phase_perturb_amount[:, None], (1, phase.shape[1]))
            perturb_mask = perturb_mask * phase_perturb_amount
            perturb_noise = torch.tile(perturb_mask[:, :, None], (1, 1, phase.shape[2]))
            phase = phase + perturb_noise
            
        elif self.perturb_phase_per_column:
            column_shape = torch.Size([phase.shape[0], phase.shape[2]])
            perturb_mask = torch.randn(column_shape, device=self.device)
            phase_perturb_amount = torch.tile(phase_perturb_amount[:, None], (1, phase.shape[2]))
            perturb_mask = perturb_mask * phase_perturb_amount
            perturb_noise = torch.tile(perturb_mask[:, None, :], (1, phase.shape[1], 1))
            phase = phase + perturb_noise
            
        perturbed_stft = magnitude * torch.exp(1j * phase)

        # Inverse STFT to obtain the perturbed audio
        perturbed_audio = torch.istft(
            perturbed_stft,
            n_fft=self.window_size,
            hop_length=self.hop_size,
            window=torch.hann_window(self.window_size).to(self.device),
            length=original_length
        )

        return perturbed_audio
        
class MultiResolutionPhasePerturber(nn.Module):
    def __init__(self, 
                 window_sizes,
                 hop_sizes,
                 perturb_setting, 
                 device):
        super(MultiResolutionPhasePerturber, self).__init__()
        self.window_sizes = window_sizes
        self.hop_sizes = hop_sizes
        assert len(window_sizes) == len(hop_sizes), "The number of window sizes and hop sizes should be the same"
        self.perturb_setting = perturb_setting
        self.device = torch.device(device)
        
        # Phase
        self.perturb_phase_per_row = perturb_setting['perturb_phase_per_row']
        self.perturb_phase_per_column = perturb_setting['perturb_phase_per_column']
        self.perturb_phase = False
        if self.perturb_phase_per_row or self.perturb_phase_per_column:
            self.perturb_phase = True
            
        self.phase_perturbers = []
        for i in range(len(window_sizes)):
            self.phase_perturbers.append(PhasePerturber(window_sizes[i], hop_sizes[i], perturb_setting, device))
        
    def forward(self, audio_tensor):
        for i in range(len(self.window_sizes)):
            perturbed = self.phase_perturbers[i](audio_tensor)
            audio_tensor[:, :perturbed.shape[1]] += perturbed
        audio_tensor = audio_tensor / len(self.window_sizes)
        return audio_tensor
    
if __name__ == "__main__":
    # Load an audio file
    audio, sr = librosa.load("test.wav", sr=None)
    audio = torch.from_numpy(audio).float().to("cuda")

    # Create the perturber
    perturber = Perturber(window_size=1024, 
                          hop_size=256, 
                          perturb_setting={
                                            'perturb_magnitude_per_row': False,
                                            'perturb_magnitude_per_column': False,
                                            'perturb_phase_per_row': False,
                                            'perturb_phase_per_column': True,
                                            'magnitude_perturb_amount': 0,
                                            'phase_perturb_amount': 1
                                        },
                          device="cuda")

    # Perturb the audio
    perturbed_audio = perturber(audio)
    perturbed_audio = perturbed_audio.cpu().numpy()
    # librosa.output.write_wav("perturbed.wav", perturbed_audio, sr)
    sf.write("perturbed.wav", perturbed_audio, sr)