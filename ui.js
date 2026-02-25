// ----------------- Global state -----------------
let schema;
let rows = [];

// ----------------- Build parameter inputs -----------------
function buildParamInputs(params){
    const container = document.getElementById("params");
    container.innerHTML = ""; // clear previous inputs

    for(const k in params){
        const spec = params[k];
        const inputType = spec.type === "int" || spec.type === "float" ? "number" : "text";

        container.innerHTML += `
            <label>${k}: 
                <input type="${inputType}" id="p_${k}" value="${spec.default}" 
                       oninput="updateDynamicMax(); render()">
            </label><br>`;
    }
}

function updateMutualMax(rowIndex){
    const probInput = document.getElementById(`p_${rowIndex}_probability`);
    const dropInput = document.getElementById(`p_${rowIndex}_drop_probability`);
    const probSlider = document.getElementById(`slider_${rowIndex}_probability`);
    const dropSlider = document.getElementById(`slider_${rowIndex}_drop_probability`);

    if(probInput && dropInput && probSlider && dropSlider){
        // Update max values based on the other field
        probInput.max = 1 - (rows[rowIndex].drop_probability ?? 0);
        dropInput.max = 1 - (rows[rowIndex].probability ?? 0);

        probSlider.max = probInput.max;
        dropSlider.max = dropInput.max;

        // Clamp current values if needed
        if(parseFloat(probInput.value) > probInput.max) probInput.value = probInput.max;
        if(parseFloat(dropInput.value) > dropInput.max) dropInput.value = dropInput.max;

        rows[rowIndex].probability = parseFloat(probInput.value);
        rows[rowIndex].drop_probability = parseFloat(dropInput.value);
    }
}




function inputHTML(spec, value, rowIndex, key){
    if(spec.type === "select"){
        let options = spec.options.map(o => `<option value="${o}" ${o===value?'selected':''}>${o}</option>`).join("");
        return `<select onchange="rows[${rowIndex}]['${key}']=this.value; render()">${options}</select>`;
    }

    if(spec.type === "float"){
        let step = spec.step !== undefined ? spec.step : 0.01;
        let min = spec.min ?? 0;
        let max = spec.max ?? 1;

        return `
        <div style="display:flex; align-items:center;">
            <!-- Number input -->
            <input type="number" id="p_${rowIndex}_${key}" value="${value}" min="${min}" max="${max}" step="${step}" 
                  style="width:60px; margin-right:5px;"
                  oninput="
                        rows[${rowIndex}]['${key}']=parseFloat(this.value);
                        document.getElementById('slider_${rowIndex}_${key}').value=this.value;
                        updateMutualMax(${rowIndex});
                        render();
                  ">

            <!-- Slider -->
            <input type="range" id="slider_${rowIndex}_${key}" min="${min}" max="${max}" step="${step}" value="${value}" 
                  style="flex:1;"
                  oninput="document.getElementById('p_${rowIndex}_${key}').value=this.value"
                  onchange="
                        rows[${rowIndex}]['${key}']=parseFloat(this.value);
                        updateMutualMax(${rowIndex});
                        render();
                  ">
        </div>`;
    }


    if(spec.type === "int"){
        let min = spec.min ?? 0;
        let max = spec.max ?? 100;
        let step = spec.step ?? 1;
        return `<input type="number" value="${value}" min="${min}" max="${max}" step="${step}" 
                       oninput="rows[${rowIndex}]['${key}']=parseInt(this.value); render()">`;
    }

    return `<input type="text" value="${value}" oninput="rows[${rowIndex}]['${key}']=this.value; render()">`;
}









function drawRows(){
    const rowsTable = document.getElementById("rows");
    rowsTable.innerHTML = "";

    rows.forEach((r, i) => {
        let tr = "<tr>";
        for (const k in r) {
            // This is the key line — always call inputHTML()
            tr += `<td>${inputHTML(schema.row[k], r[k], i, k)}</td>`;
        }
        tr += `<td><button onclick="rows.splice(${i},1); drawRows(); render()">X</button></td>`;
        tr += "</tr>";
        rowsTable.innerHTML += tr;
    });
}

function updateDynamicMax(){
    rows.forEach((row, rowIndex) => {
        for(const key in schema.row){
            const spec = schema.row[key];
            if(spec.max_param){
                const paramVal = parseFloat(document.getElementById("p_" + spec.max_param)?.value ?? 0);
                const maxVal = paramVal;

                const input = document.querySelector(`#rows tr:nth-child(${rowIndex+1}) td:nth-child(${Object.keys(row).indexOf(key)+1}) input`);
                if(input){
                    input.max = maxVal;
                    // Optional: clamp current value to new max
                    if(parseFloat(input.value) > maxVal) input.value = maxVal;
                    row[key] = parseFloat(input.value);
                }
            }
        }
    });
}



// ----------------- Collect values -----------------
function collectParams(){
    const paramsData = {};
    for(const k in schema.params){
        const el = document.getElementById("p_"+k);
        const type = schema.params[k].type;
        paramsData[k] = (type === "int" || type === "float") ? parseFloat(el.value) : el.value;
    }
    return paramsData;
}

function collectRows(){
    return rows;
}

// ----------------- Add Row -----------------
function addRow(){
    if(!schema || !schema.row) return;
    const row = {};
    for(const k in schema.row){
        row[k] = schema.row[k].default;
    }
    rows.push(row);
    drawRows();
    render();
}

// ----------------- Render SVG -----------------
async function render(){
    const data = {
        params: collectParams(),
        rows: collectRows()
    };

    try{
        const svg = await fetch("/render", {
            method:"POST",
            headers:{"Content-Type":"application/json"},
            body:JSON.stringify(data)
        }).then(r=>r.text());
        document.getElementById("preview").innerHTML = svg;
    } catch(e){
        console.error("Render error:", e);
    }
}

// ----------------- Load schema -----------------
async function loadSchema(){
    try{
        schema = await fetch("/schema").then(r=>r.json());
        console.log("Schema loaded:", schema);

        buildParamInputs(schema.params);
        drawRows();

        // enable Add Row button now
        document.getElementById("addRowBtn").disabled = false;

        // Add one starter row automatically
        addRow();
    } catch(e){
        console.error("Error loading schema:", e);
    }
}

// ----------------- Init -----------------
window.onload = loadSchema;


