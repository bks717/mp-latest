const scanner = require('./progressive_scanner.js');

async function testScan() {
    console.log('Starting scan...');
    const id = scanner.startProgressiveScan(26.9057, 93.8170);
    console.log('Scan ID:', id);

    for (let i = 0; i < 8; i++) {
        await new Promise(r => setTimeout(r, 1000));
        const s = scanner.getScanStatus(id);
        console.log(`T+${i+1}s: status=${s.status}, stage=${s.current_stage?.label}, msg="${s.status_message}", has_result=${!!s.current_result}, prov=${s.provenance_count}`);
        if (s.status === 'COMPLETED') break;
    }
    const s = scanner.getScanStatus(id);
    console.log('\nFinal status metrics:');
    console.log('Rainfall obs:', JSON.stringify(s.current_result?.metrics?.rainfall_observed_24h, null, 2));
    console.log('River Discharge:', JSON.stringify(s.current_result?.metrics?.river_discharge, null, 2));
    console.log('Elevation:', JSON.stringify(s.current_result?.metrics?.elevation, null, 2));
    console.log('Soil Moisture:', JSON.stringify(s.current_result?.metrics?.soil_moisture, null, 2));
    console.log('SAR Radar:', JSON.stringify(s.current_result?.metrics?.satellite_radar_sar, null, 2));
    console.log('Population Exposure:', JSON.stringify(s.current_result?.metrics?.population_exposure, null, 2));
    console.log('Risk formula:', s.current_result?.metrics?.risk_index?.formula_evaluation);
    console.log('Total Provenance Logs:', scanner.getScanProvenance(id).length);
}

testScan().catch(console.error);
