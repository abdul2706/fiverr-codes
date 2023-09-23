<?php

$__BR__ = '<br />';
$path_card_generator = dirname(dirname(__FILE__));
// echo $path_card_generator . $__BR__;
$path_card_images = $path_card_generator . '/images/cards';

$card_names = scandir($path_card_images);
$card_names = array_slice($card_names, 2);
// print_r($card_names);
// echo $__BR__;
$return_json = json_encode($card_names);
echo $return_json;
