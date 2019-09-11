# -*- coding: utf-8 -*-

"""Created on Mon Feb  4 17:19:03 2019

@author: Jorge C. Lucero

Computation of the vocal tract frequency response by a transmission
line analogy, following Sondhi and Schroeter, "A hybrid time-frequency
domain articulatory speech sythesizer", IEEE Trans. Acoust. Speech Signal
Proc. ASSP-35, 1987.
"""

import numpy as np
from scipy.interpolate import splev, splrep

# Constants

RHO = 0.001112             # Air density
C = 35661.                 # Speed of sound
A = 130.*np.pi             # Ratio of wall resistance to mass
B = (30.*np.pi)**2         # Squared ang freq of mechanical response
C1 = 4.                    # Correction for thermal conduct. and viscosity
W02 = (406.*np.pi)**2      # Lowest sq ang freq of acoustic resonance
DSAMPLE_POINTS = 20        # Points for downsample


class FreqResponse(object):

    def __init__(self, fup=5000, deltaf=10, downsample=True):

        """ Arguments:

        fup: max frequency
        deltaf: frequency step
        downsample: reduce number of area sections
        """

        w = 2*np.pi*np.arange(10., fup, deltaf)
        jw = (1.j)*w
        alpha = np.sqrt(jw*C1)
        beta = jw*W02/((jw + A)*jw + B) + alpha

        self.sigma_C = np.sqrt((alpha + jw)*(beta + jw))/C
        self.gamma_RC = RHO*C*np.sqrt((alpha + jw)/(beta + jw))

        self.rad_reacta = jw*8*RHO/(3*np.pi*np.sqrt(np.pi))
        self.rad_resist = 0.1*RHO*w*w/(2*np.pi*C)

        self.w = w
        self.fup = fup
        self.deltaf = deltaf

        if downsample:
            self.gross_scale = np.linspace(0, 1, DSAMPLE_POINTS)

        self.downsample = downsample

    def __repr__(self):

        return "max freq = {}, delta freq = {}".format(self.fup, self.deltaf)

    def dsample(self, y):

        tck = splrep(np.linspace(0, 1, len(y)), y, s=0)
        return splev(self.gross_scale, tck, der=0)

    def get_response(self, area, length):

        """ Arguments:

        area: areas of vocal tract sections (numpy array)
        length: total length of vocal tract
        """

        if self.downsample and len(area) > DSAMPLE_POINTS:
            area = self.dsample(area)

        ntubes = area.size
        arg = self.sigma_C*length/ntubes
        radiation_imp = self.rad_resist + self.rad_reacta/np.sqrt(area[-1])

        chain = np.zeros((self.w.size, ntubes, 2, 2), dtype=np.complex_)

        chain[:, :, 0, 0] = np.cosh(arg)[:, np.newaxis] @ np.ones((1, ntubes))
        chain[:, :, 0, 1] = ((self.gamma_RC*np.sinh(arg))[:, np.newaxis]
                             @ (1./area)[np.newaxis, :])
        chain[:, :, 1, 0] = ((np.sinh(arg)/self.gamma_RC)[:, np.newaxis]
                             @ area[np.newaxis, :])
        chain[:, :, 1, 1] = chain[:, :, 0, 0]

        transfer = chain[:, 0, :, :]

        for j in range(1, ntubes):
            transfer = np.matmul(transfer, chain[:, j, :, :])

        ug = np.ravel(transfer[:, 1, 0]*radiation_imp
                      + transfer[:, 1, 1])

        ht = 1/np.abs(ug)**2
        freq = self.w/(2*np.pi)
        formants, bw = self.get_resonances(freq, ht)
        response = 10*np.log10(ht)

        return freq, response, formants, bw

    def get_resonances(self, x, y):

        """ Arguments:

        x: frequency [Hz] (numpy array)
        y: power amplitude (numpy array)

        Returns:

        res_f:  resonances (numpy array)
        res_bw: bandwidths (numpy array)
        """

        nres = (np.diff(np.sign(np.diff(y))) < 0).nonzero()[0] + 1
        n = len(nres)
        res_f = np.zeros(n)
        res_bw = np.zeros(n)

        for i in range(n):
            ulim = min(nres[i] + 2, len(x))
            dlim = max(nres[i] - 2, 0)
            icw = np.arange(dlim, ulim + 1)
            c = np.polyfit(x[icw], 1./y[icw], 2)
            res_f[i] = -c[1]/2/c[0]
            res_q = 1./2./np.sqrt(4*c[0]*c[2]/c[1]**2-1)
            res_bw[i] = res_f[i]/res_q

        return res_f, res_bw
