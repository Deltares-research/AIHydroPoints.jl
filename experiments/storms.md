# North Sea Storm Surge Events by Year

Summary of notable storm surge events along the Dutch North Sea coast, relevant for
model training, validation, and testing. Water levels in metres above NAP unless noted.
Sources: Rijkswaterstaat stormvloedflitsen, KNMI, Wikipedia, Waddenacademie.

---

## 2010

Four stormvloedflitsen issued:

| Date | Notes |
|------|-------|
| 30 Jan | — |
| 30 Aug | Unusual summer event |
| 24 Oct | — |
| 12 Nov | Force 10 winds over Waddenzee; most significant of the year |

Overall: moderately active year, no exceptional peaks. Good as a training year.

---

## 2011

Nine stormvloedflitsen issued — very active late season, especially December.

| Date | Storm | Peak level (Delfzijl) | Notes |
|------|-------|-----------------------|-------|
| 4 Feb | — | — | — |
| 25 Nov | Cyclone Berit | — | — |
| 27–28 Nov | — | — | — |
| 3 Dec | — | — | — |
| 7–8 Dec | — | — | — |
| 6 Dec* | — | ~+4.82 m NAP* | Exceptional level; described as a 1-in-1000-year water level at Delfzijl (unconfirmed, single source) |
| 9 Dec | — | — | — |
| 16–17 Dec | — | — | — |
| 23–24 Dec | — | — | — |
| 28–30 Dec | — | — | — |

\* The ~+4.82 m NAP level at Delfzijl on 6 December 2011 would be near the all-time record
(+4.83 m NAP on 1 Nov 2006). Treat with caution — could not be confirmed by a second source.

Overall: **2011 is a much more extreme year than it appears** — the December cluster alone
makes it a very demanding validation year. Worth keeping in mind when interpreting validation
RMSE relative to training.

---

## 2012

Seven stormvloedflitsen issued — very active January, driven by the Ulli/Andrea cyclone sequence.

| Date | Storm | Peak surge | Notes |
|------|-------|-----------|-------|
| 3–4 Jan | Cyclone Ulli | — | First of the Andrea sequence |
| 5–6 Jan | — | — | — |
| 7 Jan | Cyclone Andrea | +2.30 m above normal (Hoek van Holland), +2.48 m (Rotterdam) | Flood defences closed at Den Oever, Harlingen, Delfzijl, Oosterscheldekering, Hollandse IJssel |
| 12–13 Jan | — | — | — |
| 21–22 Jan | — | — | — |
| 14–15 Feb | — | — | — |
| 9 Dec | — | — | — |

Storm Andrea (Jan 2–8) is used as a named test window in the experiment TOMLs.

---

## 2020

| Date | Storm | Peak level (Delfzijl) | Notes |
|------|-------|-----------------------|-------|
| 9–10 Feb | Ciara | ~+3.50 m NAP | Coupures Delfzijl closed; Hollandsche IJsselkering and Oosterscheldekering on standby |
| 17 Feb | Dennis | — | Less severe than Ciara |

Overall: February-dominated, comparable in magnitude to 2022.

---

## 2021

No significant storm surge events found in sources consulted.
Relatively quiet year on the Dutch coast (Storm Arwen Nov 2021 was significant for UK east coast).

---

## 2022

Exceptionally active year — multiple major storms in quick succession in February.

| Date | Storm | Peak level (Delfzijl) | Notes |
|------|-------|-----------------------|-------|
| 29–31 Jan | Corrie | — | Oosterscheldekering closed |
| ~16 Feb | Dudley | — | Part of triple storm sequence |
| 18 Feb | Eunice | +3.73 m NAP | Top 3 heaviest storms in 50+ years (KNMI); coupures Delfzijl closed; Hamburg +3.75 m surge up Elbe |
| ~21 Feb | Franklin | — | Followed immediately after Eunice |

Overall: most active recent year for surges. Eunice is the headline event.

---

## Reference events (historical)

| Date | Storm | Peak level | Location |
|------|-------|-----------|----------|
| 1 Feb 1953 | North Sea flood | — | Catastrophic; led to Delta Works |
| 27 Feb 1990 | — | +3.84 m NAP | Vlissingen |
| 28 Jan 1994 | — | +3.87 m NAP | — |
| 9 Nov 2007 | — | +3.67 m NAP | — |
| 1 Nov 2006 | Allerheiligenvloed | +4.83 m NAP | Delfzijl (highest since 1877) |
