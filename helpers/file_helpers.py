import os
from datetime import datetime


from pathlib import Path

from werkzeug.utils import secure_filename

#Save file to directory
def save_file(folder_path, file_name, file):
    is_success = False
    ex = ''

    try:
        file.save(os.path.join(folder_path, secure_filename(file_name)))
        is_success = True
    except Exception as e:
        ex = str(e)

    return is_success, ex

#Create folder after parrent folder and folder name
def create_folder(parrent_folder, folder_name):
    folder_path = parrent_folder + str(folder_name)

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    return folder_path + '/'

#Create (get) file path = folder_path/file_name_req_date_time.file_extention
def create_file_path(folder_path, file_name_req, file_extention):
    file_name = create_file_name(file_name_req, file_extention)
    file_path = folder_path + "/" + file_name
    return file_path

#Create (get) file name
def create_file_name(file_name_req, file_extention):
    now = datetime.now()
    date_time =  now.strftime("%Y-%m-%d %H-%M-%S")

    file_name = date_time + '_' + str(file_name_req) + file_extention

    return file_name