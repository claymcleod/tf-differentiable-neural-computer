var colormap = ['#ffeda0','#feb24c','#f03b20'];
var rect_size = 25;
var text_size = 18;
var offset = 5;

var colorScale = d3.scale.linear()
                .domain([0, 0.5, 1])
                .range(colormap);

function parse_msg(msg) {
    var data = [];
    for (var i = 0; i < msg.free_gates.length; i++) {
        data.push({
            "free_gates": msg.free_gates[i],
            "write_keys": msg.write_keys[i],
            "read_keys": msg.read_keys[i],
            "allocation_gates": msg.allocation_gates[i],
            "write_gates": msg.write_gates[i],
            "timestep": i
        });

    }

    return data;
}



function update_plot(data) {

    var col = 0;
    var xOffset = 145;
    var slotSize = data[0].read_keys.length
    var columnSize = slotSize * rect_size + offset;

    var yOffset = 0;
    var selection = d3.select("svg").selectAll("rect")
                .data(data);

    /** Write gates **/

    var write_gate_text = d3.select("svg").append("text")
                .attr("x", 0)
                .attr("y", yOffset+text_size)
                .text("Write gates")
                .attr("font-family", "Source Sans Pro")
                .attr("font-size", text_size)
                .attr("fill", "#000");

    selection.enter()
                .append("rect")
                .attr("width", rect_size)
                .attr("height", rect_size)
                .attr("transform", function(d) { return "translate("+(d.timestep*(offset+rect_size*slotSize) + xOffset)+", "+yOffset+")"; })
                .attr("fill", function(d) { return colorScale(d.write_gates[0][0]); })
                .attr("stroke", "#000")
                .attr("stroke-width", 1.0)

    yOffset += (offset + rect_size*2)

    /** Free gates **/

    var free_gate_text = d3.select("svg").append("text")
                .attr("x", 0)
                .attr("y", yOffset+text_size)
                .text("Free gates")
                .attr("font-family", "Source Sans Pro")
                .attr("font-size", text_size)
                .attr("fill", "#000");

    selection.enter()
                .append("rect")
                .attr("width", rect_size)
                .attr("height", rect_size)
                .attr("transform", function(d) { return "translate("+(d.timestep*(offset+rect_size*slotSize) + xOffset)+", "+yOffset+")"; })
                .attr("fill", function(d) { return colorScale(d.free_gates[0]); })
                .attr("stroke", "#000")
                .attr("stroke-width", 1.0)

    yOffset += (offset + rect_size*2)

    /** Allocation gates **/

    var allocation_gate_text = d3.select("svg").append("text")
                .attr("x", 0)
                .attr("y", yOffset+text_size)
                .text("Allocation gates")
                .attr("font-family", "Source Sans Pro")
                .attr("font-size", text_size)
                .attr("fill", "#000");

    selection.enter()
                .append("rect")
                .attr("width", rect_size)
                .attr("height", rect_size)
                .attr("transform", function(d) { return "translate("+(d.timestep*(offset+rect_size*slotSize) + xOffset)+", "+yOffset+")"; })
                .attr("fill", function(d) { return colorScale(d.allocation_gates[0][0]); })
                .attr("stroke", "#000")
                .attr("stroke-width", 1.0)
}

var socket = io();
socket.on("event", function(msg){
    console.log("Updating plot!");
    var data = parse_msg(msg);
    update_plot(data);
});
