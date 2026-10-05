# Label schema: stretcher / bed / wheelchair / cart

One criterion for every label. The criterion is **the object's shape**, not where the scene is or
the state of the patient. Reason: a model learns shape; "stretcher in a corridor, bed in a room" is not learnable.

## 0 — stretcher
A narrow, wheeled platform **built for transport**.
- Exposed tubular/X frame, the underside is visible
- Large swivel casters, usually above 15 cm
- **No solid panel** at head or foot; at most folding metal rails
- A single thin mattress, often with straps/safety belts
- Examples: ambulance cot, emergency-room trolley, transfer stretcher, wheeled incubator carrier, a field stretcher if wheeled

## 2 — bed
A bed the patient **stays** in; shape cues:
- **Solid panel** at head and/or foot (headboard/footboard)
- Motorised/articulated back section, side control panel
- Wider body, full-length rails, bedding and a pillow
- Small casters or casters hidden under a skirt
- Examples: ward bed, ICU bed, home bed, examination table (fixed, no wheels)

## Edge cases (decision table)
| Seen | Class | Reason |
|---|---|---|
| Narrow vehicle with an open frame pushed down a corridor | stretcher | shape |
| Wide bed with solid panels pushed down a corridor | bed | shape |
| Narrow vehicle with an open frame standing in a room | **stretcher** | shape, not place |
| Fixed stretcher inside an ambulance | stretcher | shape |
| Operating table (single column, no wheels) | bed | panel/table body |
| Examination table, stretcher-type with wheels | stretcher | shape |
| Incubator (transparent box) | none, clip rejected | out of schema |
| Hand-carried canvas stretcher without wheels | none, rejected | outside the door scenario |

## 1 — wheelchair
Seated position, backrest + two large rear wheels. If a person is visible but the chair is not, the box is rejected.

## 3 — cart
Supply/medication/food cart with shelves. No patient is carried on it.

## Application
- The decision is made **per clip**, but the criterion is the shape list above.
- A clip that cannot be decided is **rejected**; it is not added to a class "because it is close enough".
- Class `9` marks an **ignore region** (a partly visible transport): neither a required detection nor a false positive.
- The detector merges stretcher, wheelchair and bed into one class, `transport`; the four labels are kept for per-class reports.
