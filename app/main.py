from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse
from io import BytesIO
import aiohttp
import onnxruntime
import skimage
import warnings
import PIL
import numpy as np

app = Starlette()
session = onnxruntime.InferenceSession("/app/models/pet/pet.onnx")
data_classes = ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'boxer', 'chihuahua', 'english_cocker_spaniel', 'english_setter', 'finnish_lapphund', 'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger', 'miniature_pinscher', 'newfoundland', 'pomeranian', 'pug', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier']
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

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
                    Select image to upload:
                    <input type="file" name="file">
                    <input type="submit" value="Upload Image">
                </form>
                Or submit a URL:
                <form class="url-form" action="/classify-url" method="get">
                    <input type="url" name="url">
                    <input type="submit" value="Fetch and analyze image">
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
                };
                $('.file-form').submit(function() {
                    var formData = new FormData(this);
                    $.ajax({
                        url: $(this).attr('action'),
                        type: $(this).attr('method'),
                        data: formData,
                        success: handleResponse,
                        cache: false,
                        contentType: false,
                        processData: false
                    });
                    return false;
                });
                $('.url-form').submit(function() {
                    $.get($(this).attr("action"), $(this).serialize(), handleResponse);
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
    return predict_image_from_bytes(bytes)


@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    return predict_image_from_bytes(bytes)


def predict_image_from_bytes(bytes):
    img = my_open_image(BytesIO(bytes))
    normalized_img = my_normalize(img, np.array(imagenet_stats[0]), np.array(imagenet_stats[1]))
    numpy_input = normalized_img[None, ...]
    resized_image = skimage.transform.rescale(numpy_input, [1, 1, 244/numpy_input.shape[2], 244/numpy_input.shape[3]])

    input_name = session.get_inputs()[0].name 
    output_name = session.get_outputs()[0].name
    results = session.run([output_name], {input_name: resized_image})
    data_classes
    return JSONResponse({
        'prediction': data_classes[np.argmax(results)],
        'scores': sorted(zip(data_classes, map(float, results[0][0])),key=lambda p: p[1],reverse=True),
    })


async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()

def my_open_image(fn):
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
  return ((x-mean[...,None,None]) / std[...,None,None]).astype(np.float32, copy=False)