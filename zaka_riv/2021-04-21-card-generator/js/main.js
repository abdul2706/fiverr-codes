class CustomImage {
    constructor(img, x, y, w, h) {
        this.img = img;
        this.x = Math.round(x);
        this.y = Math.round(y);
        this.w = Math.round(w);
        this.h = Math.round(h);
        this.scale_within_bound();
    }

    scale_within_bound() {
        let ratio_w = this.w / this.img.width;
        let ratio_h = this.h / this.img.height;
        let ratio = Math.min(ratio_w, ratio_h);
        this.w = Math.round(this.img.width * ratio);
        this.h = Math.round(this.img.height * ratio);
    }

    draw() {
        imageMode(CORNER);
        image(this.img, this.x, this.y, this.w, this.h);
    }

    console_log() {
        console.log("CustomImage ['x': " + this.x +
            ", 'y': " + this.y +
            ", 'w': " + this.w +
            ", 'h': " + this.h + "]");
    }
}

class CustomText {
    constructor(text, x, y, text_color, text_size) {
        this.text = text;
        this.x = x;
        this.y = y;
        this.text_color = text_color;
        this.text_size = text_size;
    }

    draw() {
        textAlign(CENTER, CENTER);
        textSize(Number.parseInt(this.text_size));
        fill(this.text_color);
        text(this.text, this.x, this.y);
    }

    console_log() {
        console.log("CustomText ['x': " + this.x + ", 'y': " + this.y + "]");
    }
}

var TAG = '[main]';
var TAG_CANVAS = '[canvas]';
var WIDTH = 1000,
    HEIGHT = 1000;
var card_image = null,
    card_text = null;
var CARDS_INFO = [];
var CARD_SELECTED = false;
var TEXT_FONT;

function setup() {
    var card_canvas = createCanvas(WIDTH, HEIGHT);
    card_canvas.parent('card-canvas');
    TEXT_FONT = loadFont('fonts/Comfortaa-VariableFont_wght.ttf');
}

$(function () {
    // fetching image names from php script and fill card-container dynamically
    $.ajax({
        type: 'POST',
        url: 'php/cards_info.php',
        data: '',
        contentType: false,
        processData: false,
        cache: false,
        success: function (result) {
            CARDS_INFO = JSON.parse(result);
            let cards_html = '';
            for (let i = 0; i < CARDS_INFO.length; i++) {
                // for (let i = 0; i < 10; i++) {
                let card_image_path = 'images/cards/' + CARDS_INFO[0];
                cards_html += '<div class="col">';
                cards_html += '    <div class="card h-100">';
                cards_html += `        <img src="${card_image_path}" class="card-img-top" alt="${CARDS_INFO[0]}">\n`;
                cards_html += '    </div>';
                cards_html += '</div>';
            }
            $('#cards-container #cards-grid').html(cards_html);

            $('#cards-container img').click(function (e) {
                $('.selected-card').removeClass('selected-card');
                $(e.target).parent().addClass('selected-card');
                CARD_SELECTED = $(e.target).attr('alt');
            })
        },
        error: function (jqXHR, textStatus, errorThrown) {
            console.log(TAG, '[ajax error][starts]');
            console.log("Error, status = " + textStatus + ", " + "error thrown: " + errorThrown);
            console.log(TAG, '[ajax error][ends]');
        }
    });

    // save-card callback to download image on user side and save record on server side
    $('#save-card').click(function () {
        if (CARD_SELECTED) {
            card_path = 'images/cards/' + CARD_SELECTED;
            let loaded_image = loadImage(card_path, function () {
                card_image = new CustomImage(loaded_image, 0, 0, WIDTH, HEIGHT);
                card_text = new CustomText($('#name-field').val(), card_image.w * 0.5, card_image.h * 0.9, '#ffffff', 32);

                // set necessary variables
                let FINAL_WIDTH = 1024,
                    FINAL_HEIGHT = 1024;
                let ratio_w = FINAL_WIDTH / Number.parseInt(card_image.w),
                    ratio_h = FINAL_HEIGHT / Number.parseInt(card_image.h);
                let ratio = Math.min(ratio_w, ratio_h);

                // setup p5js graphics object (canvas) for drawimg image and text
                let final_canvas = createGraphics(card_image.w, card_image.h);
                final_canvas.imageMode(CORNER);
                final_canvas.rectMode(CORNER);
                final_canvas.angleMode(DEGREES);

                // draw image
                final_canvas.background(255);
                final_canvas.imageMode(CORNER);
                final_canvas.image(card_image.img, card_image.x, card_image.y, card_image.w, card_image.h);

                // draw text
                final_canvas.textAlign(CENTER, CENTER);
                final_canvas.textSize(card_text.text_size * ratio);
                final_canvas.fill(card_text.text_color);
                final_canvas.textFont(TEXT_FONT);
                final_canvas.text(card_text.text, card_text.x * ratio, card_text.y * ratio);

                // debugging part
                // card_image.console_log();
                // card_text.console_log();
                // console.log(card_text.text, card_text.x * ratio, card_text.y * ratio);

                if (card_text.text === '') {
                    card_text.text = 'empty'
                }
                let text = card_text.text.replace(' ', '-').toLowerCase();
                let image_name = text + '-' + CARD_SELECTED;

                // send data to php script for saving in csv
                let fd = new FormData();
                fd.append('access-token', 'save-card');
                fd.append('card-text', card_text.text);
                fd.append('card-selected', CARD_SELECTED);
                fd.append('image-name', image_name);
                $.ajax({
                    type: "POST",
                    url: 'php/card.php',
                    data: fd,
                    contentType: false,
                    processData: false,
                    cache: false,
                    success: function (result) {},
                    error: function (jqXHR, textStatus, errorThrown) {
                        console.log('ajax error: starts');
                        console.log("Error, status = " + textStatus + ", " + "error thrown: " + errorThrown);
                        console.log('ajax error: ends');
                    }
                });

                alert('Card Image saved as: ' + image_name);
                save(final_canvas, image_name);
            });
        } else {
            alert("Please Select Card!!!");
        }
    });

});
