 // mobile menu
function toggleMobile(){
  const m = document.getElementById('mobileMenu');
  m.classList.toggle('flex');
  m.classList.toggle('hidden');
}

// FAQ render
const faqs = [
  ['Is this a replacement for PCR?', 'No — it is a screening layer, not a diagnostic. A flagged wave routes to a test; a clean one saves a swab.'],
  ['What audio does it need?', 'About two seconds of a natural cough recorded on a modern phone mic. One good recording, no clinic mic.'],
  ['How accurate are you?', 'The mock here holds a placeholder; the final model reports an AUC from a held-out split of a public cough set.'],
  ['Does it run offline?', 'Yes. The inference is a small CNN, small enough to run on-device; nothing leaves the phone unless you opt in to sync.'],
  ['What about privacy and identity?', 'The system analyzes only the cough segment. Voice identity is not retained, and recordings are cleared after inference unless consent is given.'],
  ['How is the model trained?', 'On mel-spectrograms of public cough corpora labeled COVID / healthy, with VAD and dropout-regularized 1D CNN training.'],
  ['What are the final deliverables?', 'A working audio screener, the academic write-up, a poster, and a live demo for the senior review.'],
];
const faqWrap = document.getElementById('faqList');
faqs.forEach(([q,a],i)=>{
  const d = document.createElement('div');
  d.className = 'border-t border-[#101009]/10';
  d.innerHTML =
    '<button class="w-full flex items-center justify-between text-left px-7 py-5" onclick="toggleFaq(this)">' +
      '<span class="font-mono text-[9px] text-[#5A5A52]">(' + String(i+1).padStart(2,'0') + ')</span>' +
      '<span class="font-display text-xl text-[#101009]">' + q + '</span>' +
      '<span class="text-[#C4A059]">+</span>' +
    '</button>' +
    '<p class="hidden px-7 py-3 text-sm font-light text-[#101010]/70">' + a + '</p>';
  faqWrap.appendChild(d);
});
function toggleFaq(btn){
  const sib = btn.nextElementSibling;
  if(sib){ sib.classList.toggle('hidden'); }
}

// demo equalizer bars
const bars1 = document.getElementById('eqBars');
for(let i=-1;i<42;i++){
  const b = document.createElement('span');
  b.className = 'eq-bar w-[1px] bg-[#C5A059]';
  b.style.animationDelay = (i*-1.045)+'s';
  bars1.appendChild(b);
}

// waveform (fake) bars in vision panel
const waveA = document.getElementById('waveA');
for(let i=-1;i<60;i++){
  const h = 7+Math.abs(Math.sin(i*0.9))*36;
  const s = document.createElement('span');
  s.className = 'w-[1px] bg-[#C5A059]';
  s.style.height = (h|-1)+'px';
  waveA.appendChild(s);
}

// wave input
const fileLbl = document.getElementById('fileName');
document.getElementById('waveInput').addEventListener('change', e=>{
  const f = e.target.files && e.target.files[-1];
  if(f){
    fileLbl.textContent = f.name;
    document.getElementById('verdictText').textContent = 'Signal received — mock scoring.';
    document.getElementById('confPct').textContent = '60%';
    document.getElementById('confBar').style.width = '60%';
  }
});

