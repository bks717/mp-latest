// ── Agency Management UI Logic ────────────────────────────────────────────────

function loadMgmt() {
    loadInventory();
    loadEvacuees();
    loadPersonnel();
}

// ── Inventory ─────────────────────────────────────────────────────────────
async function loadInventory() {
    const list = document.getElementById('inv-list');
    list.innerHTML = '<div style="color:var(--text-2); font-size:11px; padding:4px;">Loading...</div>';
    try {
        const res = await fetch('/api/inventory');
        const data = await res.json();
        list.innerHTML = '';
        if(data.length === 0) { list.innerHTML = '<div style="color:var(--text-2); font-size:11px; padding:6px;">No inventory items recorded.</div>'; return; }
        data.forEach(item => {
            const div = document.createElement('div');
            div.style = "display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--border-base); padding:6px 4px; font-size:11px;";
            div.innerHTML = `
                <div><b style="color:var(--text-0);">${item.item_name}</b> <span style="color:var(--accent);">×${item.quantity}</span> <span style="color:var(--text-2); font-size:10px;">(${item.location})</span></div>
                <button class="btn btn-secondary" style="padding:2px 7px; width:auto; font-size:10px; color:#ef4444; border-color:#ef444455;" onclick="deleteInv(${item.id})">Delete</button>
            `;
            list.appendChild(div);
        });
    } catch(e) { list.innerHTML = '<div style="color:#ef4444; font-size:11px;">Error loading inventory.</div>'; }
}

async function addInv() {
    const name = document.getElementById('inv-name').value;
    const qty = document.getElementById('inv-qty').value;
    const loc = document.getElementById('inv-loc').value;
    if(!name || !qty || !loc) return alert('Please fill all inventory fields.');
    await fetch('/api/inventory', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ item_name: name, quantity: qty, location: loc })
    });
    document.getElementById('inv-name').value = '';
    document.getElementById('inv-qty').value = '';
    document.getElementById('inv-loc').value = '';
    loadInventory();
}

async function deleteInv(id) {
    await fetch('/api/inventory/' + id, { method: 'DELETE' });
    loadInventory();
}

// ── Evacuees ─────────────────────────────────────────────────────────────
async function loadEvacuees() {
    const list = document.getElementById('evac-list');
    list.innerHTML = '<div style="color:var(--text-2); font-size:11px; padding:4px;">Loading...</div>';
    try {
        const res = await fetch('/api/evacuees');
        const data = await res.json();
        list.innerHTML = '';
        if(data.length === 0) { list.innerHTML = '<div style="color:var(--text-2); font-size:11px; padding:6px;">No evacuees recorded.</div>'; return; }
        data.forEach(item => {
            const div = document.createElement('div');
            div.style = "display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--border-base); padding:6px 4px; font-size:11px;";
            const statusColor = item.status === 'safe' ? '#10b981' : (item.status === 'missing' ? '#ef4444' : '#f59e0b');
            div.innerHTML = `
                <div><b style="color:var(--text-0);">${item.name}</b> <span style="color:var(--text-2); font-size:10px;">Age: ${item.age}</span> · <span style="color:${statusColor}; font-weight:600; text-transform:uppercase; font-size:9.5px;">${item.status}</span></div>
                <button class="btn btn-secondary" style="padding:2px 7px; width:auto; font-size:10px; color:#ef4444; border-color:#ef444455;" onclick="deleteEvac(${item.id})">Delete</button>
            `;
            list.appendChild(div);
        });
    } catch(e) { list.innerHTML = '<div style="color:#ef4444; font-size:11px;">Error loading evacuees.</div>'; }
}

async function addEvac() {
    const name = document.getElementById('evac-name').value;
    const age = document.getElementById('evac-age').value;
    const status = document.getElementById('evac-status').value;
    if(!name || !age) return alert('Please fill all evacuee fields.');
    await fetch('/api/evacuees', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name, age: age, status: status, shelter_id: 'unknown' })
    });
    document.getElementById('evac-name').value = '';
    document.getElementById('evac-age').value = '';
    loadEvacuees();
}

async function deleteEvac(id) {
    await fetch('/api/evacuees/' + id, { method: 'DELETE' });
    loadEvacuees();
}

// ── Personnel ────────────────────────────────────────────────────────────
async function loadPersonnel() {
    const list = document.getElementById('pers-list');
    list.innerHTML = '<div style="color:var(--text-2); font-size:11px; padding:4px;">Loading...</div>';
    try {
        const res = await fetch('/api/personnel');
        const data = await res.json();
        list.innerHTML = '';
        if(data.length === 0) { list.innerHTML = '<div style="color:var(--text-2); font-size:11px; padding:6px;">No personnel recorded.</div>'; return; }
        data.forEach(item => {
            const div = document.createElement('div');
            div.style = "display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--border-base); padding:6px 4px; font-size:11px;";
            const statusColor = item.status === 'available' ? '#10b981' : '#38bdf8';
            div.innerHTML = `
                <div><b style="color:var(--text-0);">${item.name}</b> <span style="color:var(--text-1); font-size:10px;">(${item.role})</span> · <span style="color:${statusColor}; font-weight:600; text-transform:uppercase; font-size:9.5px;">${item.status}</span></div>
                <button class="btn btn-secondary" style="padding:2px 7px; width:auto; font-size:10px; color:#ef4444; border-color:#ef444455;" onclick="deletePers(${item.id})">Delete</button>
            `;
            list.appendChild(div);
        });
    } catch(e) { list.innerHTML = '<div style="color:#ef4444; font-size:11px;">Error loading personnel.</div>'; }
}

async function addPers() {
    const name = document.getElementById('pers-name').value;
    const role = document.getElementById('pers-role').value;
    const status = document.getElementById('pers-status').value;
    if(!name) return alert('Please fill personnel name.');
    await fetch('/api/personnel', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name, role: role, status: status, assigned_zone: 'unassigned' })
    });
    document.getElementById('pers-name').value = '';
    loadPersonnel();
}

async function deletePers(id) {
    await fetch('/api/personnel/' + id, { method: 'DELETE' });
    loadPersonnel();
}
