from starlette.applications import Starlette
from starlette.responses import JSONResponse
from fastai.vision import *
import aiohttp

async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()

app = Starlette()

@app.route("/")
async def homepage(request):
    return JSONResponse({"message": "Hello World!"})

pet_learner = load_learner('/app/models/pet')

@app.route("/classify-pet", methods=["GET"])
async def classify_pet(request):
    bytes = await get_bytes(request.query_params["url"])
    img = open_image(BytesIO(bytes))
    _,_,losses = pet_learner.predict(img)
    return JSONResponse({
        "predictions": sorted(
            zip(pet_learner.data.classes, map(float, losses)),
            key=lambda p: p[1],
            reverse=True
        )
    })