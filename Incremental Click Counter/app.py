from flask import Flask, render_template

app = Flask(__name__)

skor = 0

@app.route('/')
def index():
    return render_template('index.html', skor=skor)

@app.route('/tambah')
def tambah():
    global skor
    skor += 1
    return render_template('index.html', skor=skor)

@app.route('/kurangi')
def kurangi():
    global skor
    skor -= 1
    return render_template('index.html', skor=skor)

if __name__ == '__main__':
    app.run(debug=True)