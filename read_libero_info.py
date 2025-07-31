import h5py

f = h5py.File("libero_90_converted/meta_info.h5", "r")
for key in f.keys():
    print(f[key].name)  #获得名称，相当于字典中的key
