import os
import cv2
import pytesseract
import pandas as pd
from PyPDF2 import PdfReader

def check_for_words(value):
    value = value.lower()
    return (
           'signature' in value \
        or 'grantee' in value \
        or 'buyer' in value \
        or 'owner' in value \
        # or 'approval' in value \
    )
    # return ('signature' in value) and (
    #        'grantee' in value \
    #     or 'buyer' in value \
    #     or 'owner' in value \
    #     # or 'approval' in value \
    # )

download_dir = os.path.abspath('./downloaded_pdfs')
print('[download_dir]', download_dir)
cropped_images_dir = os.path.abspath('./cropped_images')
print('[cropped_images_dir]', cropped_images_dir)
os.makedirs(cropped_images_dir, exist_ok=True)

temp_image_path = 'temp.jpg'
pdfs_downloaded = os.listdir(download_dir)[2:]
print('[pdfs_downloaded]', pdfs_downloaded)

for pdf_idx, pdf_filename in enumerate(pdfs_downloaded):
    pdf_path = os.path.join(download_dir, pdf_filename)
    reader = PdfReader(pdf_path)
    page = reader.pages[0]
    print(pdf_filename, '[reader.pages]', len(reader.pages), '[page.images]', len(page.images))
    for img_idx, image_file_object in enumerate(page.images):
        # temp_image_path = f'{pdf_filename}-{img_idx}-{image_file_object.name}'
        # print('[temp_image_path]', temp_image_path)
        with open(temp_image_path, "wb") as fp:
            fp.write(image_file_object.data)

        image = cv2.imread(temp_image_path)
        image = cv2.GaussianBlur(image, ksize=(25, 25), sigmaX=1.0)
        H, W = image.shape[0], image.shape[1]
        df = pytesseract.image_to_data(image, output_type='data.frame')
        df['right'] = df['left'] + df['width']
        df['bottom'] = df['top'] + df['height']
        df = df[['block_num', 'line_num', 'left', 'top', 'right', 'bottom', 'width', 'height', 'text']]
        df = df.loc[~df['text'].isna()].reset_index(drop=True)
        # print(pdf_filename, img_idx, '[df]\n', df)
        df2 = pd.DataFrame(columns=df.columns)
        for grp_id, grp in df.groupby(['block_num', 'line_num']):
            # print(grp_id, grp)
            # print(grp_id, ' '.join(grp.text.values))
            width = grp['right'].max() - grp['left'].min()
            height = grp['bottom'].max() - grp['top'].min()
            if width >= 0.5 * W or height >= 0.5 * H: continue
            text = ' '.join(grp.text.values)
            if len(text) < 5: continue
            new_row = pd.DataFrame({'block_num': grp_id[0], 
                       'line_num': grp_id[1], 
                       'left': grp['left'].min(), 
                       'top': grp['top'].min(), 
                       'right': grp['right'].max(), 
                       'bottom': grp['bottom'].max(), 
                       'width': width, 
                       'height': height, 
                       'text': text}, index=[0])
            df2 = pd.concat([df2, new_row], ignore_index=True)
        print(pdf_filename, img_idx, '[df2]\n', df2)

        df3 = df2.loc[df2['text'].apply(check_for_words)].reset_index(drop=True)
        print(pdf_filename, img_idx, '[df3]\n', df3)

        # exit()
        # signatures_block_num = df3.loc[df3['text'].str.find('Signature') >= 0, 'block_num']
        # grantees_block_num = df3.loc[df3['text'].str.find('Grantee') >= 0, 'block_num']
        # print(pdf_filename, img_idx, '[signatures_block_num]', signatures_block_num.values)
        # print(pdf_filename, img_idx, '[grantees_block_num]', grantees_block_num.values)
        # common_block_nums = set(signatures_block_num).intersection(set(grantees_block_num))
        # print(pdf_filename, img_idx, '[common_block_nums]', common_block_nums)
        # df3 = df3.loc[df3['block_num'].isin(common_block_nums)].reset_index(drop=True)
        if len(df3) > 0:
            # df3['right'] = df3['left'] + df3['width']
            # df3['bottom'] = df3['top'] + df3['height']
            # bbox = [df3['left'].min(), df3['top'].min(), df3['right'].max(), df3['bottom'].max()]
            # bbox_width, bbox_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            # large_bbox = [bbox[0] - bbox_width, bbox[1], bbox[2] + bbox_width, bbox[3] + 2*bbox_height]
            for i, row in df3.iterrows():
                (x, y, w, h) = (row['left'], row['top'], row['width'], row['height'])
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)
            # cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
            # cv2.rectangle(image, (large_bbox[0], large_bbox[1]), (large_bbox[2], large_bbox[3]), (255, 0, 0), 3)
            
            # cv2.imwrite(crop_image_path, image)
            image = cv2.resize(image, dsize=None, fx=0.20, fy=0.20)
            cv2.imshow('image', image)
            cv2.waitKey(0)

            # # image = cv2.imread(temp_image_path)
            # # image = cv2.GaussianBlur(image, ksize=(25, 25), sigmaX=1.0)
            # # _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            # image_crop = image[large_bbox[1]:large_bbox[3], large_bbox[0]:large_bbox[2]]
            # crop_image_path = os.path.join(cropped_images_dir, f'{pdf_filename}-{img_idx}.jpg')
            # cv2.imwrite(crop_image_path, image_crop)
        print('*' * 50)
    break

os.remove(temp_image_path)
