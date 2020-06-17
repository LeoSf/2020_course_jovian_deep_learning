# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Biong. Leandro D. Medus
Ph.D Student GPDD - ETSE
Universitat de València
leandro.d.medus@uv.es

12-06-2020

Script Description:
-------------------
    Basic functionalities to test if the device is cuda capable and analyzed the total amount of memory being used

Version: 1.0
------------

"""

# import torch
# print(torch.cuda.is_available())

import torch
import pycuda.driver as cuda

#import pycuda.autoinit # Necessary for using its aboutCudaDevices

class aboutCudaDevices():
    def __init__(self):
        pass

    def num_devices(self):
        """Return number of devices connected."""
        return cuda.Device.count()

    def devices(self):
        """Get info on all devices connected."""
        num = cuda.Device.count()
        print("%d device(s) found:" % num)
        for i in range(num):
            print(cuda.Device(i).name(), "(Id: %d)" % i)

    def mem_info(self):
        """Get available and total memory of all devices."""
        available, total = cuda.mem_get_info()
        print("Available: %.2f GB\nTotal:     %.2f GB" % (available / 1e9, total / 1e9))

    def attributes(self, device_id=0):
        """Get attributes of device with device Id = device_id"""
        return cuda.Device(device_id).get_attributes()

    def __repr__(self):
        """Class representation as number of devices connected and about them."""
        num = cuda.Device.count()
        string = ""
        string += ("%d device(s) found:\n" % num)
        for i in range(num):
            string += ("    %d) %s (Id: %d)\n" % ((i + 1), cuda.Device(i).name(), i))
            string += ("          Memory: %.2f GB\n" % (cuda.Device(i).total_memory() / 1e9))
        return string


def info_memory():
    """
    To get current usage of memory you can use pyTorch's functions such as:
    :return:
    """
    import torch
    # Returns the current GPU memory usage by
    # tensors in bytes for a given device
    print("torch.cuda.memory_allocated(): ", torch.cuda.memory_allocated())

    # Returns the current GPU memory managed by the
    # caching allocator in bytes for a given device
    print("torch.cuda.memory_cached(): ", torch.cuda.memory_cached())


def clear_cache():
    """
    Releases all unoccupied cached memory currently held by
    the caching allocator so that those can be used in other
    GPU application and visible in nvidia-smi
    :return:
    """
    torch.cuda.empty_cache()


if __name__ == '__main__':

    print("torch.cuda.is_available():\t\t", torch.cuda.is_available())

    cuda.init()

    # Get Id of default device
    print("torch.cuda.current_device():\t", torch.cuda.current_device())        # output: 0

    # '0' is the id of your GPU
    print("cuda.Device(0).name():\t\t\t" + cuda.Device(0).name())               # output: GeForce GTX 1050 Ti

    print("torch.cuda.get_device_name(0):\t" + torch.cuda.get_device_name(0))   # output: GeForce GTX 1050 Ti

    # You can print output just by typing its name (__repr__):
    print(aboutCudaDevices())
    # 1 device(s) found:
    #     1) GeForce GTX 1050 Ti(Id: 0)
    #         Memory: 4.29 GB

    # To get current usage of memory you can use pyTorch's functions such as:
    info_memory()

    # And after you have run your application, you can clear your cache using a simple command:
    clear_cache()

