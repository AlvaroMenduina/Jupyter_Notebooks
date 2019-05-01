import os
import shutil

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


    # Change the name of a existing Directory
    #os.rename(new_path, new_name)

    # ==========================================================================================
    ### Creating files

    with open('dogs.txt', 'w') as file:
        names = ['Max', 'Cooper', 'Pepper', 'Bentley', 'Hunter']
        for n in names:
            file.write(n)
            file.write('\n')
        file.close()

    ### Moving and Renaming files

    # Make a new directory New Dogs
    new_name = os.path.join(base, 'New Dogs')
    try:
        os.mkdir(new_name)
    except FileExistsError:
        pass

    # Move the file dogs.txt from \Dogs to \New Dogs
    new_cwd = os.getcwd()
    old_file = os.path.join(new_cwd, 'dogs.txt')
    new_file = os.path.join(new_name, 'dogs_moved.txt')
    shutil.move(old_file, new_file)

    # Move back to the original Directory where os_methods.py is
    os.chdir(cwd)

    # ==========================================================================================
    ### Listing the files in the Directory
    file_list = os.listdir()
    print('\nListing the files in Directory: ', cwd)
    for f in file_list:
        print(f)
        if f.endswith('.txt'):
            print('This is a text file')

    # ==========================================================================================
    ### Walking a director tree
    # Move to the Main Directory (Jupyter Notebooks)
    os.chdir(base)
    for dir_name, folders, files in os.walk(os.getcwd()):
        print('The current directory is ' + dir_name)

        print('\n==============================================')
        for folder in folders:
            print('Folder: ' +  folder)

        for file in files:
            print('File: ' + file)

    walk_dir = list(os.walk(os.getcwd()))
    print('\nWalking the Directory :' + os.getcwd())
    print('Length of Walk: ', len(walk_dir))
    print('First layer Walk[0] :', walk_dir[0])
    print('\nWalk[0][0]: Directory Name -->', walk_dir[0][0])
    print('Walk[0][1]: List of Folders -->', walk_dir[0][1])
    print('Walk[0][2]: Files in Directory -->', walk_dir[0][2])










