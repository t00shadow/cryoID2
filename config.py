# store default config
import configparser

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config['DEFAULT'] = {

    }
    with open('config.ini', 'w') as configfile:
        config.write(configfile)
