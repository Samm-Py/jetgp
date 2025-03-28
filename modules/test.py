import modules.primes as pr
import numpy as np
import modules.vdc as vdc
import modules.halton as halton
import modules.primes as primes
import modules.hammersley as hs
import time
# a = pr.create_primes(100000000)
# a = a[a!=0]

# a = np.arange(0,1e8,1, dtype = 'int32')

# a = vdc.create_van_der_corput_samples(a, 2)

test = pr.create_primes(2)

st = time.time()
b = halton.create_halton_samples(1e3, 1,0, test)
et = time.time()

tt = et - st

print(tt)

# c = hs.create_hammersley_samples(1e3,2,0,test)

