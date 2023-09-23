<?php

$csv_file_name = 'data.csv';
$columns = array('datetime', 'image_name', 'card_selected', 'card_text');

if ($_SERVER['REQUEST_METHOD'] == 'POST') {
    if (isset($_POST['access-token'])) {
        $token = $_POST['access-token'];
        if ($token == 'save-card') {
            $card_text = $_POST['card-text'];
            $card_selected = $_POST['card-selected'];
            $image_name = $_POST['image-name'];
            if (!file_exists($csv_file_name)) {
                $csv_file = fopen('data.csv', 'w');
                fputcsv($csv_file, $columns, $delimiter = ',');
                fclose($csv_file);
            }
            $csv_file = fopen('data.csv', 'a');
            $data = array(date("Y-m-d"), $image_name, $card_selected, $card_text);
            fputcsv($csv_file, $data, $delimiter = ',');
            fclose($csv_file);
        }
    }
}
echo 'file accessed directly going back';
