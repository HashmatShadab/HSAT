import os
import shutil


if __name__ == "__main__":

    # write code for looping over all the .tgz files in the directory
    # and extracting the contents of the .tgz files in the same location
    # as the .tgz file and then deleting the .tgz file

    # get the working directory
    working_directory = r"D:\hidisc_data\studies"
    compressed_directory = r"D:\hidisc_data\compressed"

    # create the compressed directory if it does not exist
    if not os.path.exists(compressed_directory):
        os.makedirs(compressed_directory)

    # loop over all the files in the working directory
    for file in os.listdir(working_directory):
        # check if the file is a .tgz file
        print(file)
        if file.endswith(".tgz"):
            # extract the contents of the .tgz file
            os.system(f"tar -xvzf {os.path.join(working_directory, file)} -C {working_directory}")
           # move the .tgz file to the compressed directory
            shutil.move(os.path.join(working_directory, file), compressed_directory)
            # delete the .tgz file
