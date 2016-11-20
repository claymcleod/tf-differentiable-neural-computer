var fs = require("fs");
var express = require('express');
var app = express();
var http = require('http').Server(app);
var io = require('socket.io')(http);

if (process.argv.length !== 3) {
    console.error("Usage: node "+process.argv[1]+" file_to_watch");
    process.exit(1);
}

app.use(express.static('static'))

function read_summary_json() {
    return JSON.parse(fs.readFileSync(process.argv[2]));
}

io.on('connection', function(socket){
    try {
        json = read_summary_json();
        socket.emit("event", json);
    } catch (err) {}
    
    fs.watch('../data/summary.json', (eventType, filename) => {
      try {
          json = read_summary_json();
          socket.emit("event", json);
      } catch (err) {}
    });
});

http.listen(3003, function(){
    console.log('Serving on port 3003...');
});
