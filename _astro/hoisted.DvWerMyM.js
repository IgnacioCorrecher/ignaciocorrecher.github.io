import"./hoisted.Bl07r5t7.js";document.addEventListener("DOMContentLoaded",()=>{const n=document.getElementById("searchBar"),o=document.getElementById("tagSelector"),a=document.querySelectorAll(".grid-item");function e(){const r=n.value.toLowerCase(),c=o.value.toLowerCase();a.forEach(t=>{const s=t.querySelector(".title");if(s){const l=s.textContent?.toLowerCase()||"",d=(t.getAttribute("data-tags")||"").toLowerCase().split(","),i=l.includes(r),m=c===""||d.includes(c);t.style.display=i&&m?"flex":"none"}})}n.addEventListener("input",e),o.addEventListener("change",e),e()});
