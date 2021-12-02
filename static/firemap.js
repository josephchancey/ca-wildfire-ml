// set the map center
var firemap = L.map("firemap", {
    center: [37.573, -121, 494],
    zoom: 6
});

// add the tile layer
baseLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
})
baseLayer.addTo(firemap);

// ['#ffffb2','#fecc5c','#fd8d3c','#f03b20','#bd0026']
function getColor(duration) {
    if (parseInt(duration) > 100) {
        return "#CE2029";
    }
    else if (parseInt(duration) > 75) {
        return "#D84528";
    }
    else if (parseInt(duration) > 50) {
        return "#E26928";
    }
    else if (parseInt(duration) > 25) {
        return "#EB8E27";
    }
    else {
        return "#F5B227";
    }
}



// load fire data
d3.json(`/inactive/fires`).then(function (fires) {

    console.log(fires);
    var fireLayer = new L.LayerGroup()
    var fireLayer2013 = new L.LayerGroup()
    var fireLayer2014 = new L.LayerGroup()
    var fireLayer2015 = new L.LayerGroup()
    var fireLayer2016 = new L.LayerGroup()
    var fireLayer2017 = new L.LayerGroup()
    var fireLayer2018 = new L.LayerGroup()
    var fireLayer2019 = new L.LayerGroup()
    var fireLayer2020 = new L.LayerGroup()
    var fireLayer2021 = new L.LayerGroup()

    // var fireArray = fires.map(fire => [fire.Latitude,fire.Longitude])
    fires.forEach(fire => {
        fireMarker = L.circle(
            [fire.Latitude, fire.Longitude], {
            fillColor: getColor(fire["Duration(Days)"]),
            color: getColor(fire["Duration(Days)"]),
            fillOpacity: .5,
            opacity: 1,
            radius: fire.AcresBurned * 0.3
        }).bindPopup("<b>Fire: </b>" + fire.Name + "</br>" + "<b>Acres Burned: </b>" + fire.AcresBurned + 
        "</br>" + "<b>Duration: </b>" + fire["Duration(Days)"] + " days" + 
        "<img align='right' src='https://emojipedia-us.s3.amazonaws.com/source/skype/289/fire_1f525.png' width='40' height='40'>" + "</br>" + "</br>")

        // var year = fire.StartedDateOnly.split("-")[0]
        if (fire.Year == "2013") {
            fireMarker.addTo(fireLayer2013)
        }
        if (fire.Year == "2014") {
            fireMarker.addTo(fireLayer2014)
        }
        if (fire.Year == "2015") {
            fireMarker.addTo(fireLayer2015)
        }
        if (fire.Year == "2016") {
            fireMarker.addTo(fireLayer2016)
        }
        if (fire.Year == "2017") {
            fireMarker.addTo(fireLayer2017)
        }
        if (fire.Year == "2018") {
            fireMarker.addTo(fireLayer2018)
        }
        if (fire.Year == "2019") {
            fireMarker.addTo(fireLayer2019)
        }
        if (fire.Year == "2020") {
            fireMarker.addTo(fireLayer2020)
        }
        if (fire.Year == "2021") {
            fireMarker.addTo(fireLayer2021)
        }

        fireMarker.addTo(fireLayer)
    });


    d3.json(`/active/fires`).then(function (fires2) {

        console.log(fires2);
        var fireLayerActive = new L.LayerGroup()

        fires2.forEach(fire => {
            fireMarker = L.circle(
                [fire.Latitude, fire.Longitude], {
                fillColor: "red",
                color: "red",
                fillOpacity: .5,
                opacity: 1,
                radius: 1
            }).bindPopup("<b>Fire: </b>" + fire.Name + "</br>" + "<b>Acres Burned: </b>still active" + "</br>" + "<b>Duration: </b>unknown" + 
            "<img align='right' src='https://emojipedia-us.s3.amazonaws.com/source/skype/289/fire_1f525.png' width='40' height='40'>" + "</br>" + "</br>")
            fireMarker.addTo(fireLayerActive)
        });


        var overlays = {
            "Historic Fires": fireLayer,
            "2013": fireLayer2013,
            "2014": fireLayer2014,
            "2015": fireLayer2015,
            "2016": fireLayer2016,
            "2017": fireLayer2017,
            "2018": fireLayer2018,
            "2019": fireLayer2019,
            "2020": fireLayer2020,
            "2021": fireLayer2021,
            "Active Fires": fireLayerActive
        }

        L
            .control
            .layers({"OpenStreetMap": baseLayer}, overlays, {
                collapsed: false,
                sortLayers: true
            })
            .addTo(firemap)

        fireLayer.addTo(firemap)

    });
});