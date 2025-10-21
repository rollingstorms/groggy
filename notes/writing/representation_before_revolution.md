
# Representation Before Revolution: A New Cartography of Thought

*By Mike Roth*


## I. Representation Before Revolution

Every revolution in human understanding begins not with discovery, but with a **new representation**—a new way to make the invisible visible.  Somehow, when we change our view of the world, our capacity to think expands without any biological upgrade to the brain.  The neurons stay the same; the geometry of understanding changes.

When **Ptolemy** drew his first global maps, he gave sailors a new way to think about the Earth: the curved made flat, the [endless made navigable] (https://en.wikipedia.org/wiki/Ptolemy's_world_map#:~:text=The%20Ptolemy%20world%20map%20is,credited%20to%20Agathodaemon%20of%20Alexandria).  When **René Descartes** placed algebra on a grid, [geometry](https://en.wikipedia.org/wiki/La_G%C3%A9om%C3%A9trie#:~:text=Mathematical%20appendix%20to%20Descartes%27%20Discourse,on%20Method%2C%20published%20in%201637) became something we could compute, visualize, and reason about symbolically.  When **Gottfried Wilhelm Leibniz** (alongside Newton) developed [calculus](https://en.wikipedia.org/wiki/History_of_calculus#:~:text=Calculus%20%2C%20originally%20called%20,have%20continued%20to%20the%20present), motion became a language of symbols—differentiation and integration transforming the continuous into steps of thought.  When **John von Neumann** proposed the [stored‑program computer]9https://en.wikipedia.org/wiki/Von_Neumann_architecture#:~:text=A%20stored,to%20route%20data%20and%20control), data and instruction—once separate worlds—merged into a single flow of memory.  And when **Claude Shannon** [described](https://en.wikipedia.org/wiki/A_Mathematical_Theory_of_Communication#:~:text=,9) information as bits, entropy became mathematics: the shape of uncertainty quantified.

Each moment redefined what could be seen, not just what could be built.  It was as if a hidden axis of the universe had rotated into view.

**John Conway’s Game of Life** (1970) was the next turning point—a visual proof of Turing’s dream that one simple machine could perform every computation imaginable.  Out of four trivial rules—birth, death, survival, and stasis—emerged gliders, replicators and entire ecologies.  Complexity arose from nothing.  A plane of pixels became a living simulation.  The [Game of Life](https://en.wikipedia.org/wiki/Conway's_Game_of_Life#:~:text=The%20Game%20of%20Life%2C%20also,or%20any%20other%20Turing%20machine) is a **cellular automaton** devised by Conway in 1970; it is a zero‑player game whose evolution is determined by its initial state and is known to be Turing complete.  Its popularity comes from the surprising ways in which patterns evolve, providing an example of emergence and self‑organization.  For the first time, computation contained its own physics.

That is the rhythm of progress: **representation before revolution**.  First, we draw the world differently, and almost immediately, we understand it differently.  When the lens changes, everything downstream transforms.

![PtolemyWorldMap.jpg](img/PtolemyWorldMap.jpg)

*A mid‑15th‑century Florentine map of the world based on Ptolemy’s “Geography”, an early example of mapping the curved earth onto a rectangular projection.*

---

## II. The Universe as Computation

Modern physics is quietly rediscovering what computer scientists have long suspected: the universe itself might be **computational**.  In *A New Kind of Science* (NKS), **Stephen Wolfram** [imagines space](https://www.wolframscience.com/nks/p475--space-as-a-network/) as a **causal network**.  Each particle, field, and ray of light is a stable pattern in an evolving graph.  Reality is not a smooth continuum but a web of discrete relationships updating in parallel.  The thesis of [NKS](https://en.wikipedia.org/wiki/A_New_Kind_of_Science#:~:text=The%20thesis%20of%20A%20New,2) is twofold: the nature of computation must be explored experimentally, and the results of these experiments have great relevance to understanding the physical world.  Wolfram calls the simple systems he studies “simple programs” and argues that the scientific philosophy and methods appropriate for them apply widely.

What we perceive as continuity—motion, fields, waves—is the coarse‑grained projection of that deeper discreteness.  Each node of the network carries its own perspective, its own local clock, its own center of the universe.  From that vantage, “space” isn’t emptiness but adjacency; “time” isn’t flow but update order.

If that’s true, then graphs are not metaphors for the world—they **are** its substrate.  Every molecule, neuron, or photon is simply a local configuration of relationships.  What we call physics may be nothing more than computation unfolding at cosmic scale, light traveling through a network of rules that slightly distort its path with each hop.

To say “the universe computes itself” is not mystical—it’s pragmatic.  Computation is just the evolution of relationships through rules.  The cosmos may be the largest discrete program ever executed, and we are inside the runtime.

![rule30_pattern.png](img/rule30_pattern.png)

*A pattern generated by **Rule 30**, one of the simplest cellular automata studied by Wolfram.  Even though the rules are extremely simple, the pattern displays complex, seemingly random behavior—an example of emergence in simple programs.*

---

## III. The Modern Fracture: Data Science as 21st‑Century Cartography

So what does any of this mean for the people who write code?

In our own century, the new mapmakers are data scientists.  We chart invisible territories—social networks, molecular structures, financial systems, neural embeddings.  But our instruments are fragmented.

We have **tables**, where everything is rectangular and efficient.  We have **graphs**, where everything is relational.  We have **tensors**, where everything is differentiable and continuous.  Each is a projection of the same landscape; each distorts reality to preserve one useful property.

Tables preserve clarity.  Graphs preserve structure.  Tensors preserve flow.  None preserve the whole.

The result is an endless translation loop: Pandas → PyTorch → SQL → Spark → Neo4j, edge lists → adjacency matrices → dataframes.  Context evaporates; schema breaks; meaning thins out.

We are like early cartographers using different projections of the same globe—Mercator for navigation, Gall‑Peters for area, Robinson for beauty—each true and false at once.  A map is not the territory; it’s an abstraction that makes the territory thinkable.  So it is with data: we flatten the world so that we can compute on it.

Maybe the missing piece isn’t another format or library, but a **common projection**—a structure where all these views can coexist and translate without loss.  That was the motivation behind **Groggy**: to build a minimal, fast, understandable substrate where tables, graphs, and matrices are not competing representations but interchangeable faces of the same object.

![glider.png](img/glider.png)

*A “glider” pattern from Conway’s Game of Life.  Small patterns like this move across the grid, illustrating how local rules can produce emergent behavior.*

---

## IV. Groggy: A Practical Beginning

[Groggy](https://github.com/rollingstorms/groggy) didn’t begin as a manifesto.  It began as a dare:

> 

*How simple and fast can a dynamic graph engine be?*

The answer led to a deceptively [minimal architecture](https://rollingstorms.github.io/groggy/concepts/origins/#the-ultralight-example):

- **GraphSpace** – the living state of the graph, tracking which nodes and edges exist.

- **GraphPool** – a columnar memory arena that stores attributes separately from structure, because computers think faster when data of the same kind sits together.

- **HistoryForest** – a versioned log of change, a “git for graphs,” preserving every edit as a temporal branch.

From these ideas—state, storage, and history—emerges a system that can mutate safely, recall quickly, and compute efficiently.  Everything in Groggy is abstracted: nodes point to attributes, edges point to attributes, and attributes live independently.

The design is mechanical, but the effect is philosophical: **structure and signal coexist without entanglement**.

Why columnar storage?  Not because physics is columnar—because **computers** are.  This is one of our chosen abstractions, the deliberate loss that makes speed possible.  Computers work best when memory is contiguous, when operations vectorize.  By storing graphs in columnar form, we can treat them simultaneously as tables and as networks: the rectangular and the relational united in one memory layout.

Groggy doesn’t claim to reinvent mathematics; it just tightens a loose joint between two existing worlds.  That joint—the moment when graph and table become interchangeable—is small, but it changes everything downstream.

---

## V. Structural Computing in Practice

On the surface, Groggy looks ordinary:

```python
g = Graph.from_table(transactions)
gt = g.edges.filter('amount > 500').connected_components().table()
```

But beneath that one‑liner, something subtle happens: there is **no translation**.  The table, the graph, and the matrix are the same object viewed from different angles.

That means an exploratory data‑analysis notebook and a production inference pipeline can literally share the same code.  The analyst’s experiment becomes the engineer’s deployment without rewriting.  Exploration becomes simulation; simulation becomes production.

This is what I call **structural computing**—the idea that all forms of computation, from EDA to deep learning, are just different projections of the same underlying structure.

Groggy isn’t a grand theory; it’s infrastructure for thought.  It’s what happens when you accept the constraints of current hardware—Python for conversation, Rust for performance—and build the simplest bridge possible between them.  The simplicity *is* the philosophy.

---

## VI. The Era of Vibecoding

Groggy itself was built in collaboration with AI coding assistants—Claude, ChatGPT, Codex, others—an experiment in what might be the first generation of **vibecoders**: humans designing architectures while agents fill in the assembly.

**Andrej Karpathy** recently noted that the [hottest new programming language](https://blog.almaer.com/english-will-become-the-most-popular-development-language-in-6-years/#:~:text=When%20I%20say%20%E2%80%9CEnglish%E2%80%9D%2C%20I,such%20as%20Spanish%20and%20Mandarin) is not in code, but in English, pointing out that computers can now understand natural language and write code based on it.  Programming is shifting from syntax to system design.  The human defines the shape of an idea; the model materializes it in code.  It’s the same transformation architecture underwent a century ago: craftsmen stopped laying bricks and started designing skylines.

The next generation of engineers won’t “write programs”; they’ll **compose systems of thought**.  Vibecoding isn’t about automation; it’s about collaboration—building infrastructures that are both conceptual and executable.

**Lev Manovich**, in *The Language of New Media*, distinguished narrative—a sequence determined by the author—from database, where the user creates their own path through a field of possibilities.  As Manovich [observes](https://ccdigitalpress.org/book/stories/chapters/ulman/intro_narrative.html#:~:text=,%28225), “as a cultural form, the database represents the world as a list of items, and it refuses to order this list. In contrast, a narrative creates a cause‑and‑effect trajectory of seemingly unordered items. Therefore, database and narrative are natural enemies”.  Programs are databases of potential narrative: structures within which meaning is traversed, not dictated.

By that definition, Groggy is a **narrative machine**.  It’s a database of relationships where every user writes a different story.  It belongs to a lineage that includes the *I Ching*, tarot, and the web itself—systems where structure encodes imagination.

![e0643ed0-17e6-43a6-8f2d-82e99a864308.png](img/e0643ed0-17e6-43a6-8f2d-82e99a864308.png)

*A conceptual diagram illustrating the evolution from traditional, handwritten software (Software 1.0) through data‑driven machine learning (Software 2.0) to natural‑language‑driven “vibe‑coding” (Software 3.0).*

---

## VII. Games, Graphs, and Emergence

Across mathematics and biology, complexity arises from connection.  **Conway’s Game of Life**, **Nash’s equilibria**, genetic algorithms, reinforcement learning and transformers all share one law: entities update one another through local rules.

Neural networks are weighted games of life, learning to stabilize patterns across edges.  Backpropagation is negotiation between nodes.  Graphs are where equilibrium lives.

You can’t remove a node without leaving a trace; every signal leaves an imprint.  When we decompose a graph into its signal and frequency spaces, every component still influences every other—just like in physical systems.

Groggy’s design mirrors that process.  It’s dynamic, reversible, differentiable—a small computational microcosm built to model systems that learn and evolve.  It’s suited for fraud networks adapting daily, molecular graphs shifting under simulation, or social systems re‑wiring in real time.

The point isn’t that Groggy can do all of these today.  The point is that the architecture is *ready* for them.  Fast recall, immutability, and separation of structure from signal are exactly what future computation will require.  It’s the skeleton of an organism we’re still evolving.

---

## VIII. Representation, Infrastructure, and Imagination

To visualize every molecule of a tree or every photon in a beam of light will take more than GPUs and models; it will take **organizational clarity**.  We don’t yet have a universal graph database language that can describe complex, living systems simply and consistently.  We need data structures optimized not just for speed but for meaning—for movement between scales of abstraction without friction.

Groggy is one small piece of that foundation: a tool to navigate between idea and computation, between what we can imagine and what we can run.  If it succeeds, it will be because it is simple enough to share—because it respects the limits of current machines while keeping the door open for what comes next.

This isn’t about reinventing the graph; it’s about refining the **interface between imagination and computation**.  Every generation inherits tools that shape what can be imagined.  The next generation’s imagination will be **graph‑native**.

---

## IX. The Call to Builders

We stand again at the edge of a representational shift.  The last century gave us symbolic, imperative and functional programming.  This one will give us **structural computing**—systems that treat relationships as primitives and change as first‑class.

To reach it, we need infrastructure: engines that are fast enough, simple enough, and open enough for collective exploration.  The coming revolution won’t happen in models but in **memory**—in how we store, share, and traverse the structures of thought.  Just as cartographers once mapped the oceans to connect continents, we are mapping invisible oceans of data to connect disciplines.

The task is not to build smarter algorithms, but a **common geometry of understanding**.

Groggy is only one vessel among many, but it sails in that direction—toward a world where exploration and computation are the same act, where every dataset is a map, and every graph is a conversation between minds, human and machine alike, about the shape of reality itself.

In the spirit of Nadia Eghbal’s **[Working in Public](https://press.stripe.com/working-in-public#:~:text=Over%20the%20last%2020%20years%2C,value%20of%20online%20content%20today)**, which looks at the shift in open‑source software from public collaboration to constant maintenance by often unseen solo operators and examines how examining *who* produces things online helps us understand the value of online content, we invite you not just to use Groggy but to build with it—to join a community of graph‑native cartographers mapping the computational universe.

---

## References

- **Ptolemy’s maps.**  The Ptolemy world map is a reconstruction of the world known to Greco‑Roman society in the 2nd century and was based on Ptolemy’s *Geography*.  It introduced longitudinal and latitudinal lines, revolutionizing European cartography. [en.wikipedia.org](https://en.wikipedia.org/wiki/Ptolemy's_world_map#:~:text=The%20Ptolemy%20world%20map%20is,credited%20to%20Agathodaemon%20of%20Alexandria)

- **Descartes and analytic geometry.**  *La Géométrie*, a mathematical appendix to Descartes’ *Discourse on Method* published in 1637, established the foundation of analytic geometry. [en.wikipedia.org](https://en.wikipedia.org/wiki/La_G%C3%A9om%C3%A9trie#:~:text=Mathematical%20appendix%20to%20Descartes%27%20Discourse,on%20Method%2C%20published%20in%201637)

- **Leibniz and calculus.**  Infinitesimal calculus was developed independently by Newton and Leibniz in the late 17th century, leading to the Newton–Leibniz controversy. [en.wikipedia.org](https://en.wikipedia.org/wiki/History_of_calculus#:~:text=Calculus%20%2C%20originally%20called%20,have%20continued%20to%20the%20present)

- **Von Neumann architecture.**  A stored‑program computer encodes both program instructions and data using the same mechanism; this advancement allowed computers to store instructions in memory and superseded manually rewired machines. [en.wikipedia.org](https://en.wikipedia.org/wiki/Von_Neumann_architecture#:~:text=A%20stored,to%20route%20data%20and%20control)

- **Shannon’s information theory.**  Claude Shannon’s 1948 article “A Mathematical Theory of Communication” introduced the concept of the **bit** and gave rise to the field of information theory.  The paper is one of the most influential scientific works of the 20th century. [en.wikipedia.org](https://en.wikipedia.org/wiki/A_Mathematical_Theory_of_Communication#:~:text=,9)

- **Conway’s Game of Life.**  Conway’s Game of Life, devised in 1970, is a zero‑player cellular automaton whose evolution is determined by its initial configuration.  It is Turing complete, and its patterns—gliders, guns, breeders—show how simple rules can lead to complex behavior.  The game illustrates emergence and self‑organization. [en.wikipedia.org](https://en.wikipedia.org/wiki/Conway's_Game_of_Life)

- **A New Kind of Science.**  Stephen Wolfram’s *A New Kind of Science* (2002) argues that simple programs like cellular automata can exhibit complex behavior and that experimental exploration of computation has broad implications for understanding the physical world. [https://www.wolframscience.com/nks/](https://www.wolframscience.com/nks/)

- **Karpathy on English as code.**  In January 2023, Andrej Karpathy wrote that “the hottest new programming language is English,” highlighting a shift toward natural‑language programming. [blog.almaer.com](https://blog.almaer.com/english-will-become-the-most-popular-development-language-in-6-years/#:~:text=When%20I%20say%20%E2%80%9CEnglish%E2%80%9D%2C%20I,such%20as%20Spanish%20and%20Mandarin)

- **Narrative vs. database.**  Lev Manovich observes that databases represent the world as unordered lists, whereas narratives create cause‑and‑effect trajectories; he calls them “natural enemies”. [mitpress.mit.edu](https://mitpress.mit.edu/9780262632553/the-language-of-new-media/)

- **Working in Public.**  Nadia Eghbal’s *Working in Public* notes that open‑source software has shifted from an optimistic model of public collaboration to ongoing maintenance by often unseen solo operators, emphasising that examining who produces things online helps us understand the value of online content. [press.stripe.com](https://press.stripe.com/working-in-public)

