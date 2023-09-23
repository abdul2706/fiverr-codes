import os
import pandas as pd

gift_files_dir = 'gift_output-excel_to_gift'
os.makedirs(gift_files_dir, exist_ok=True)
excel_files_dir = 'excel_output-docx_to_excel'
excel_files = os.listdir(excel_files_dir)

for i, filename in enumerate(excel_files):
    excel_filepath = os.path.join(excel_files_dir, filename)
    df = pd.read_excel(excel_filepath)
    print('[filename]', filename)

    gift_filepath = os.path.join(gift_files_dir, filename[:-5] + '.txt')
    with open(gift_filepath, 'w') as text_file:
        for i, row in df.iterrows():
            statement = row['statement']
            options = row[['option_a', 'option_b', 'option_c', 'option_d']]
            correct_option = row['answer']
            if str(correct_option).lower().strip() in ['a', 'b', 'c', 'd']:
                correct_option = row[f'option_{str(correct_option).lower().strip()}']
            wrong_options = list(filter(lambda x: x != correct_option, options))

            text_file.write(':: Question (MC) ::' + '\n')
            text_file.write(str(statement).strip() + '\t{')
            for wrong_option in wrong_options:
                text_file.write('~' + str(wrong_option).strip() + ' \t')
            text_file.write('\t=' + str(correct_option).strip() + ' \t')
            text_file.write('####' + str(row['explanation']).strip() + '}\n\n')
