from app.routes import app

if __name__ == "__main__":
    app.run(debug=True)

app.secret_key = "ulasCheck123"