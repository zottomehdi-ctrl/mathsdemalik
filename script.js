/* Onglets Cours / Fiche / Exercices */
document.addEventListener("click", function (e) {
  var tab = e.target.closest(".tab");
  if (!tab) return;
  var target = tab.dataset.target;
  document.querySelectorAll(".tab").forEach(function (t) {
    t.classList.toggle("active", t === tab);
  });
  document.querySelectorAll(".panel").forEach(function (p) {
    p.classList.toggle("active", p.id === target);
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
  if (history.replaceState) history.replaceState(null, "", "#" + target);
});

/* Ouverture directe d'un onglet via l'ancre (#fiche, #exos) */
window.addEventListener("DOMContentLoaded", function () {
  var hash = (location.hash || "").replace("#", "");
  if (hash && document.getElementById(hash)) {
    var btn = document.querySelector('.tab[data-target="' + hash + '"]');
    if (btn) btn.click();
  }
});
