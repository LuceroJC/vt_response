# -*- coding: utf-8 -*-

"""Created on Mon Feb  4 17:19:03 2019

@author: Jorge C. Lucero

Test script for responsevt.py
"""

import numpy as np
import vtresponse as vtr
import matplotlib.pyplot as plt


# Create FeqResponse object

vt = vtr.FreqResponse(5000, 10)

# Vowel /a/, data from Story et al., 1996.

area = np.array([
    .45, .20, .26, .21, .32, .30, .33, 1.05, 1.12, .85, .63, .39, .26,
    .28, .23, .32, .29, .28, .40, .66, 1.20, 1.05, 1.62, 2.09, 2.56,
    2.78, 2.86, 3.02, 3.75, 4.60, 5.09, 6.02, 6.55, 6.29, 6.27, 5.94,
    5.28, 4.70, 3.87, 4.13, 4.25, 4.27, 4.69, 5.03
    ])

# Get response

f, resp, formants, bw = vt.get_response(area, 17)

# Print formants

print('')
print('Acoustic response for vowel /a/ (in Hz):')
print('')
for i in range(len(formants)):
    print('Formant {:d} = {:5.0f}, bandwidth = {:5.0f}'.format(i + 1,
          formants[i], bw[i]))

# Plot

plt.figure(1)
plt.plot(f, resp.T)
plt.title('Acoustic response for vowel /a/')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (dB)')
plt.show()
