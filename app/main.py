from fastapi import FastAPI, Query
from app.recommender import get_content_recommendations, get_collab_recommendations

app = FastAPI(title="📚 Book Recommender API")

@app.get("/")
def root():
    return {"message": "✅ Book Recommender API is running."}

@app.get("/recommend/content")
def content_recommend(title: str = Query(..., description="Book title to base recommendations on")):
    results = get_content_recommendations(title)
    return {"recommendations": results}

@app.get("/recommend/collab")
def collab_recommend(user_id: int = Query(..., description="User ID to get personalized book recommendations")):
    results = get_collab_recommendations(user_id)
    return {"recommendations": results}