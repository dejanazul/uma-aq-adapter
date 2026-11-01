import sys

from forget_data2 import main

SOURCES = {
    'omc25_val': (
        'https://dl.fbaipublicfiles.com/opencatalystproject/data/omc/250802/omc_val_250802.tar.gz',
        8_110_386_232,
    ),
    'odac25_val_filtered': (
        'https://dl.fbaipublicfiles.com/dac/odac25/20250918/odac25_filtered_val.tar.gz',
        11_117_673_531,
    ),
}

if __name__ == '__main__':
    main(sys.argv[1], SOURCES, 'forget_data3.json')
