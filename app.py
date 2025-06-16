from app import create_app

app = create_app()
app.secret_key = "ulascheck"

if __name__ == "__main__":
    app.run(debug=True)