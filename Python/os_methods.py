import os

if __name__ == '__main__':

    # Get Current Working Directory
    cwd = os.getcwd()
    print('\nCurrent Working Directoy is: ', cwd)

    # Get Basename (the last folder of the CWD)
    base_name = os.path.basename(cwd)
    print('\nBasename is: ', base_name)

    # Split the cwd
    path = os.path.split(cwd)
    base = path[0]

    # check if path with an incorrect name exists
    incorrect_path = os.path.join(base, 'Pythonn')
    print('\nDoes ' + incorrect_path + ' exist?')
    print(os.path.exists(incorrect_path))

    # check if a directory exists, if not, make it
    new_path = os.path.join(base, 'Dogs')
    try:
        # Try to change directory to the non-existent path
        os.chdir(new_path)
    except:
        print('\nDirectory does not exist')
        print('Creating Directory: ', new_path)
        os.mkdir(new_path)

    new_name = os.path.join(base, 'New Dogs')
    # Change the name of a existing Directory
    os.rename(new_path, new_name)

    # os.rename(new_name, new_path)



