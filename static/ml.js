// Load Data
d3.json(`/inactive/fires`).then(function (data) {
    console.log(data)
});

d3.json(`/active/fires`).then(function (data) {
    console.log(data)
});

var layout = {
    title: {
        text: 'Active California Fire Start-Date',
        font: {
            family: 'Courier New, monospace',
            size: 24
        },
        xref: 'paper',
        x: 0.05,
    },
    xaxis: {
        title: {
            text: 'Fire (name)',
            font: {
                family: 'Courier New, monospace',
                size: 18,
                color: '#7f7f7f'
            }
        },
    }
};

// ping route & Store data
d3.json(`/active/fires`).then(function (myData) {
    console.log("Init now")
    let data = [
        {
            x: myData.map(x => x["Name"]),
            y: myData.map(x => x["Started"]),
            mode: 'markers',
            type: 'scatter'
        }
    ]
    console.log("let data = is now logged")
    // draw plot
    console.log("Plot should be drawn")
    Plotly.newPlot('test_stat', data, layout);
    
});

// Render plot
d3.json(`/active/fires`).then(function (myData) {
    console.log("We're in the active fires plot")
    // map values
    let newX = myData.map(x => x["Name"]);
    let newY = myData.map(x => x["Started"]);

    // restyle existing type plot
    Plotly.restyle('test_stat', 'x', [newX]);
    Plotly.restyle('test_stat', 'y', [newY]);

});

//
//
//

var layout2 = {
    title: {
        text: 'California Fire Total Acres Burned',
        font: {
            family: 'Courier New, monospace',
            size: 24
        },
        xref: 'paper',
        x: 0.05,
    },
    xaxis: {
        title: {
            text: 'Fire (Name)',
            font: {
                family: 'Courier New, monospace',
                size: 18,
                color: '#7f7f7f'
            }
        },
    },
    yaxis: {
        title: {
            text: 'Acres Burned (acres)',
            font: {
                family: 'Courier New, monospace',
                size: 18,
                color: '#7f7f7f'
            }
        }
    }
};

// ping route & Store data
d3.json(`/inactive/fires`).then(function (myData) {
    console.log("Init now")
    let data = [
        {
            x: myData.map(x => x.Name),
            y: myData.map(x => x.AcresBurned),
            mode: 'markers',
            type: 'scatter'
        }
    ]
    console.log("let data = is now logged")
    // draw plot
    console.log("Plot should be drawn")
    Plotly.newPlot('test_stat_two', data, layout2);
    
});

// Render plot
d3.json(`/inactive/fires`).then(function (myData) {
    console.log("We're in the active fires plot")
    // map values
    let newX = myData.map(x => x.Name);
    let newY = myData.map(x => x.AcresBurned);

    // restyle existing type plot
    Plotly.restyle('test_stat_two', 'x', [newX]);
    Plotly.restyle('test_stat_two', 'y', [newY]);

});
