from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse
from io import BytesIO
import aiohttp
import onnxruntime
import skimage
import warnings
import PIL
import numpy as np
from typing import List


app = Starlette()

imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

class Model:
    session: onnxruntime.InferenceSession
    classes: List[str]
    def __init__(self, session: onnxruntime.InferenceSession, classes: List[str]) -> None:
        self.session = session
        self.classes = classes

models = {
    'pet': Model(
        onnxruntime.InferenceSession("/app/models/pet.onnx"),
        ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'boxer', 'chihuahua', 'english_cocker_spaniel', 'english_setter', 'finnish_lapphund', 'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger', 'miniature_pinscher', 'newfoundland', 'pomeranian', 'pug', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier']
    ),
    'flower': Model(
        onnxruntime.InferenceSession("/app/models/flower.onnx"),
        ['Alpine Sea Holly', 'Anthurium', 'Artichoke', 'Azalea', 'Ball Moss', 'Balloon Flower', 'Barbeton Daisy', 'Bearded Iris', 'Bee Balm', 'Bird of Paradise', 'Bishop of Llandaff', 'Black-eyed Susan', 'Blackberry Lily', 'Blanket Flower', 'Bolero Deep Blue', 'Bougainvillea', 'Bromelia', 'Buttercup', 'Californian Poppy', 'Camellia', 'Canna Lily', 'Canterbury Bells', 'Cape Flower', 'Carnation', 'Cautleya Spicata', 'Clematis', "Colt's Foot", 'Columbine', 'Common Dandelion', 'Corn Poppy', 'Cyclamen', 'Daffodil', 'Desert-rose', 'English Marigold', 'Fire Lily (Glory Lily)', 'Foxglove', 'Frangipani', 'Fritillary', 'Garden Phlox', 'Gaura', 'Gazania', 'Geranium', 'Giant White Arum Lily', 'Globe Thistle', 'Globe-Flower', 'Grape Hyacinth', 'Great Masterwort', 'Hard-leaved Pocket Orchid', 'Hibiscus', 'Hippeastrum', 'Japanese Anemone', 'King Protea', 'Lenten Rose', 'Lotus', 'Love in the Mist', 'Magnolia', 'Mallow', 'Marigold', 'Mexican Aster', 'Mexican Petunia', 'Monkshood', 'Moon Orchid', 'Morning Glory', 'Orange Dahlia', 'Osteospermum', 'Oxeye Daisy', 'Passion Flower', 'Pelargonium', 'Peruvian Lily', 'Petunia', 'Pincushion Flower', 'Pink Primrose', 'Pink-yellow Dahlia', 'Poinsettia', 'Primula', "Prince of Wales' Feathers", 'Purple Coneflower', 'Red Ginger', 'Rose', 'Ruby-lipped Cattleya', 'Siam Tulip', 'Silverbush', 'Snapdragon', 'Spear Thistle', 'Spring Crocus', 'Stemless Gentian', 'Sunflower', 'Sweet Pea', 'Sweet William', 'Sword Lily', 'Thorn Apple', 'Tiger Lily', 'Toad Lily', 'Tree Mallow', 'Tree Poppy', 'Trumpet Creeper', 'Wallflower', 'Water Lily', 'Watercress', 'Wild Pansy', 'Windflower', 'Yellow Iris']
    )
}


@app.route("/")
def form(request):
    return HTMLResponse("""
        <html>
        <head>
            <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/css/materialize.min.css">
            <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
        </head>
        <body>
            <nav>
                <div class="nav-wrapper">
                    <a href="#" class="brand-logo center"><i class="large material-icons">location_searching</i>ML Classifier</a>
                </div>
            </nav>
            <div class="section flow-text">
                <form class="file-form" action="/upload" method="post" enctype="multipart/form-data">
                    
                    Which model to use?
                    <p class="flow-text">
                        <label>
                            <input name="model" type="radio" value="pet" checked />
                            <span>Cat/Dog Breeds</span>
                        </label>
                    </p>
                    <p class="flow-text">
                        <label>
                            <input name="model" type="radio" value="flower" />
                            <span>Flower Species</span>
                        </label>
                    </p>

                    Select image to upload:
                    <div class="file-field input-field">
                        <div class="btn">
                            <span>File</span>
                            <input type="file" name="file">
                        </div>
                        <div class="file-path-wrapper">
                            <input class="file-path validate" type="text">
                        </div>
                    </div>

                    Or submit a URL:
                    <input class="url-input" type="url" name="url">
                    
                    <button class="btn waves-effect waves-light" type="submit" name="action">
                        Classify     <i class="material-icons right">tune</i>
                    </button>

                </form>
            </div>
            <div class="divider"></div>
            <div class="section flow-text">
                <div class="output"></div>
            </div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/materialize/1.0.0/js/materialize.min.js"></script>
            <script src="https://code.jquery.com/jquery-3.4.1.min.js"></script>
            <script type="text/javascript">
                var handleResponse = function (response) {
                    var out = "<b>Prediction:</b> " + response.prediction + "<br><table><tr><th>Category</th><th>Score</th></tr>";
                    for (var i = 0; i < response.scores.length; i++) {
                        out += "<tr><td>" + response.scores[i][0] + "</td><td>" + response.scores[i][1] + "</td></tr>";
                    }
                    out += "</table>";
                    $('.output').html(out);
                    $('.btn').removeClass('disabled');
                    $('.file-form').trigger('reset');
                };
                $('.file-form').submit(function() {
                    var formData = new FormData(this);
                    if ($('.url-input').val().length > 0) {
                        $.get('/classify-url', $(this).serialize(), handleResponse);
                    } else {
                        $.ajax({
                            url: $(this).attr('action'),
                            type: $(this).attr('method'),
                            data: formData,
                            success: handleResponse,
                            cache: false,
                            contentType: false,
                            processData: false
                        });
                    }
                    $('.output').html('<div class="progress"><div class="indeterminate"></div></div>')
                    $('.btn').addClass('disabled');
                    return false;
                });
            </script>
        </body>
        </html>
    """)


@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes, data["model"])


@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    return predict_image_from_bytes(bytes, request.query_params["model"])


def predict_image_from_bytes(bytes, model_name: str):
    img = my_open_image(BytesIO(bytes))
    normalized_img = my_normalize(img, np.array(imagenet_stats[0]), np.array(imagenet_stats[1]))
    numpy_input = normalized_img[None, ...]
    resized_image = resize_image(numpy_input)

    model = models[model_name]
    input_name = model.session.get_inputs()[0].name 
    output_name = model.session.get_outputs()[0].name
    results = model.session.run([output_name], {input_name: resized_image})
    return JSONResponse({
        'prediction': model.classes[np.argmax(results)],
        'scores': sorted(zip(model.classes, map(float, results[0][0])),key=lambda p: p[1],reverse=True),
    })


async def get_bytes(url: str):
    "asynchronously download bytes at url"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


def my_open_image(fn):
    "open image and convert it to numpy array of float32, representing each pixel as 3 channels of 0.0 - 1.0 for RGB, based on fastai open_image"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning) # EXIF warning from TiffPlugin
        x = PIL.Image.open(fn).convert('RGB')
    a = np.asarray(x)
    if a.ndim==2 : a = np.expand_dims(a,2)
    a = np.transpose(a, (1, 0, 2))
    a = np.transpose(a, (2, 1, 0))
    x = a.astype(np.float32, copy=False)
    x = x / 255
    return x


def my_normalize(x, mean, std):
    "normalize data by subtracting mean and dividing by stddev"
    return ((x-mean[...,None,None]) / std[...,None,None]).astype(np.float32, copy=False)


def resize_image(numpy_input: np.array, rows = 224, cols = 224):
    return skimage.transform.rescale(numpy_input, [1, 1, rows/numpy_input.shape[2], cols/numpy_input.shape[3]], multichannel=False)