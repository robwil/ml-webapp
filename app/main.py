from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse
from fastai.vision import *
import aiohttp

app = Starlette()
pet_learner = load_learner('/app/models/pet')


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
    img = open_image(BytesIO(bytes))
    classes = pet_learner.data.classes
    _, class_, losses = pet_learner.predict(img)
    return JSONResponse({
        "prediction": classes[class_.item()],
        "scores": sorted(
            zip(classes, map(float, losses)),
            key=lambda p: p[1],
            reverse=True
        )
    })


async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()
