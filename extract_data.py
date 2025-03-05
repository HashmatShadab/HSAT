import os
import sys


# function to extract file using tar command
def extract_data(file_path, extract_path):
    os.system('tar -xvf ' + file_path + ' -C ' + extract_path)


if __name__ == '__main__':
    data_root = r"D:\val"
    val= [
        "NIO_003",
        "NIO_004",
        "NIO_007",
        "NIO_009",
        "NIO_010",
        "NIO_016",
        "NIO_027",
        "NIO_033",
        "NIO_034",
        "NIO_044",
        "NIO_069",
        "NIO_083",
        "NIO_085",
        "NIO_088",
        "NIO_095",
        "NIO_097",
        "NIO_100",
        "NIO_102",
        "NIO_110",
        "NIO_112",
        "NIO_116",
        "NIO_118",
        "NIO_126",
        "NIO_127",
        "NIO_137",
        "NIO_141",
        "NIO_146",
        "NIO_148",
        "NIO_164",
        "NIO_165",
        "NIO_175",
        "NIO_177",
        "NIO_181",
        "NIO_182",
        "NIO_184",
        "NIO_189",
        "NIO_193",
        "NIO_195",
        "NIO_214",
        "NIO_216",
        "NIO_219",
        "NIO_222",
        "NIO_231",
        "NIO_247",
        "NIO_251",
        "NIO_254",
        "NIO_258",
        "NIO_259",
        "NIO_261",
        "NIO_264",
        "NIO_266",
        "NIO_267",
        "NIO_274",
        "NIO_281",
        "NIO_292",
        "NIO_295",
        "NIO_296",
        "NIO_299",
        "NIO_306",
        "NIO_307"
    ]
    # loop through all the files in the data_root directory
    for file in os.listdir(data_root):
        # check if the file is a tar file
        if file.endswith('.tgz'):
            # extract the file
            print(f"Extracting {file}")
            extract_data(os.path.join(data_root, file), data_root)
            print(f"Extracted {file}")
            # Remove the tar file
            os.remove(os.path.join(data_root, file))
        else:
            print(f"File {file} is not a tgz file")