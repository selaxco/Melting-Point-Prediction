# Previsão da temperatura de <i>melting</i> de proteínas com um domínio por meio de redes neurais

<h2 align="left">Introdução</h2>
<blockquote> 
<p align="justify">Saudações! Sejam bem-vindos ao nosso projeto. :smiley_cat:</p>
<p align="justify">Nosso grupo é formado por <a href="https://github.com/gustavercosa">Gustavo Duarte Verçosa</a>, <a href="https://github.com/jpab2004">João Pedro Aroucha de Brito</a>, <a href="https://github.com/selaxco">Thaynara Beatriz Selasco de Matos</a> e <a href="https://github.com/viyuetuki">Vitória Yumi Uetuki Nicoleti</a>. Somos alunos de Ciência e Tecnologia, curso este que está sendo feito na <a href="https://ilum.cnpem.br/">Ilum Escola de Ciência</a>, faculdade do <a href="https://cnpem.br/">Centro Nacional de Pesquisa em Energia e Materiais (CNPEM)</a>.</p>
<p align="justify">Este projeto surgiu como nossa apresentação final para a disciplina de <b>Redes Neurais e Algoritmos Genéticos</b>, ministrada pelo professor doutor <a href="https://github.com/drcassar">Daniel Roberto Cassar</a>. Com a temática de incorporar conceitos biológicos, decidimos explorar uma característica físico-química das proteínas: <b>a temperatura de <i>melting</i></b>. Motivados por essa perspectiva, optamos por utilizar redes neurais para realizar a previsão da temperatura de <i>melting</i> de proteínas com um domínio.</p> 
<img src="https://img.shields.io/badge/STATUS-Em%20desenvolvimento-576CFB"> <img src="https://img.shields.io/badge/LICENCE-GNU%20General%20Public%20License%20v3.0-75CA75">
</blockquote> 

<h3 align="left">O que é temperatura de <i>melting</i>? :fire:</h3>
<blockquote> 
<p align="justify"> </p>
</blockquote>

<h3 align="left">Por que usar redes neurais? 👩🏻‍💻</h3>
<blockquote> 
<p align="justify"> </p>
</blockquote>

<h2 align="left">Banco de dados</h2>
<blockquote> 
<p align="justify"> </p>
</blockquote> 

<h2 align="left">Metodologia</h2>
<blockquote> 
<p align="justify"> Como explicado anteriormente, o projeto visa utilizar Redes Neurais como um possível método de determinação da temperatura de melting de porteínas de um único dominio, porém, precisamo primeiramente entender como essas redes neurais foram construidas e quais as possíveis alterações que podem ser feitas nos modelos. As redes neurais desenvolvidas no projeto possuem uma estrutura similar entre si e podem ser consideradas como muito parecidas, justamente por conta de um dos objetivos ser a comparação de diferentes implementações e os resultados obtidos com tais.</p>
<p align="justify">A primeira etapa para estrutração da rede foi pensar nas camadas escondidas (hidden layers) que irão compor as redes. Chegou-se na conclusão que poderiam ser utilizadas redes com 3 camadas escondidades que poderiam ter seus tamanhos variáveis a fim de comparação da eficácia e diferenciação destas.</p>
<p align="justify">Começou-se então a etapa de análise das features a serem fornecidas para a rede. Os dados obtidos através do banco de dados exposto anteriormente explicitam a presença de algumas possibilidades para esses dados de entrada (input) para as redes, porém, foi feita uma determinação destes dados, sendo consideradas as facilidades de obtenção de tais dados em conjunto com a disponibilidade de acesso de tais informações. Com esta análise, chegamos a idealização de 21 ou 22 dados de entrada que seriam alimentados a rede em busca da temperatura de melting da proteína originadora dos dados.</p>
<p align="justify">Os primeiros 20 dados a serem considerados são de facil acesso, principalmente em plataformas como o banco de dados online de proteínas <a href='https://www.uniprot.org'>UniProt</a> e se trata da quantização dos aminoácidos presentes na cadeia da proteína, ou seja, a rede receberia a quantidade de vezes que cada aminoácido se encontra na proteína. A escolha desta informação foi feita pela simplicidade e escalabilidade da rede para proteínas de diferentes tamanho sem que seja alterada sua estrutura básica.</p>
<p align="justify">Os outros dados a serem considerados são, na verdade, apenas um dado, mas expresso de 2 diferentes formas. Uma das informações obtidas no banco de dados é justamente os aditivos presentes em solução com as proteínas estudadas, levando a uma alteração nas temperaturas de melting de tais proteínas pela interação destas com os aditivos. Pensou-se então em uma forma de passar estes dados não numéricos as redes que precisam de dados numéricos. Com a análise cuidadosa, foi possível a obtenção de dois métodos de trasnformação dos dados em números. O primeiro método consiste na determinação de um peso para cada uma das possibilidades de aditivos que seriam então somados e dividios pelo total dos pesos, gerando um valor de ponto flutuante entre 0 e 1. Um exemplo desta aplicação pode ser visto abaixo, onde o aditivo 1 possui um peso de 1 e o aditivo 2 possui um peso de 2.</p>

<center>
  
| Aditivo 1 | Aditivo 2 | Total (soma / soma dos pesos) |
| :------------: | :------------: | :------------: |
| 0 | 0 | 0 |
| 1 | 0 | 0.33333 |
| 0 | 1 | 0.66666 |
| 1 | 1 | 1 |
  
</center>

<p align="justify"> O segundo método de trasnformação dos dados seria uma simples separação dos dados em dois valores binários, o primeiro valor numérico representaria a presença ou não do aditivo 1, enquanto o segundo representaria a presença ou não do aditivo 2. Os dois métodos foram utilizados para a criação das redes, desta forma, é possível a construção de redes que podem ser alteradas de diferentes formas para comparação de eficácia. </p>

<p align="justify">Ao todo foram utilizadas 6 possíveis combinações diferentes de métodos, sendo estas combinações atingidas pela utilização de 2 formas diferentes de input para o dados de aditivos e outras 3 funções de ativação que foram escolhidas para serem aplicadas ao problema.</p>
<p align="justify">Vamos ver um pouco sobre cada um dessas diferenças aplicadas </p>
</blockquote> 

<h3 align="left">Comparações :eyes:</h3>

<blockquote> 
<p align="justify">As funções de ativação desempenham um papel essencial nas redes neurais, sendo responsáveis por influenciar a saída de um neurônio. 
Neste contexto, foram testadas três funções de ativação: a <a href="https://paperswithcode.com/method/sigmoid-activation">Sigmoid Activation</a>, a <a href="https://paperswithcode.com/method/leaky-relu">Leaky ReLU</a> e a <a href="https://paperswithcode.com/method/swish">Swish</a>.</p>
  <blockquote> 
  <p align="justify">A função de ativação sigmóide é uma função não linear que mapeia valores de entrada para um intervalo entre 0 e 1. É amplamente utilizada em problemas de classificação binária, onde a saída é 0 ou 1. A função sigmóide possui um gradiente suave, o que facilita a otimização usando o gradiente descendente. No entanto, ela também apresenta algumas desvantagens, como o problema do gradiente que desaparece, o que pode tornar o treinamento de redes neurais profundas mais desafiador.</p>
  <p align="justify">A função de ativação Leaky ReLU é uma variante da função ReLU que aborda o problema conhecido como "morte ReLU". Ela introduz uma pequena inclinação para valores negativos, permitindo que o neurônio tenha uma saída diferente de zero mesmo quando a entrada é negativa. Essa inclinação geralmente é definida como um valor pequeno, como 0.01. Embora o Leaky ReLU seja computacionalmente mais exigente em comparação com o ReLU, ele pode ajudar a evitar o problema do gradiente que desaparece, que é comum em redes neurais profundas.</p>
  <p align="justify">A função de ativação Swish é uma função relativamente nova que ganhou popularidade nos últimos anos. Ela é suave e não monotônica, semelhante à função sigmoide. O Swish é definido como a multiplicação da entrada pelo resultado da função sigmoide aplicada à entrada. Essa função inclui um parâmetro $\beta$ que pode ser aprendido. O Swish demonstrou superar o ReLU e outras funções de ativação em alguns casos, mas é computacionalmente mais exigente em comparação com o ReLU.</p>
  </blockquote>
<p align="justify">Em geral, a seleção da função de ativação depende do problema específico e da arquitetura da rede neural.</p>
</blockquote>

<h2 align="left">Conclusão</h2>
<blockquote> 
<p align="justify">A conclusão do nosso projeto será apresentada durante uma aula no dia 22 de junho de 2023, no período da tarde. Após uma semana dessa apresentação, esta seção será reestruturada e atualizada aqui.</p>
</blockquote> 

<h2 align="left">Agradecimentos</h2>
<blockquote> 
<p align="justify">Gostaríamos de expressar nossos sinceros agradecimentos, em especial ao professor doutor Daniel Roberto Cassar, por sua valiosa contribuição ao explorar a disciplina conosco e discutir a elaboração do projeto.</p>
<p align="justify">Além disso, gostaríamos de agradecer à professora doutora Juliana Helena Costa Smetana por seu interesse e auxílio no entendimento dos aspectos biológicos do projeto. Sua participação foi fundamental para o desenvolvimento do nosso trabalho.</p>
</blockquote> 
  
<h2 align="left">Referências</h2>
<blockquote> 
<p align="justify">[1] </p>
<p align="justify">[2] TALAPATI, Sumalatha Rani; GOYAL, Megha; NATARAJ, Vijayashankar; <i>et al</i>. Structural and binding studies of cyclin‐dependent kinase 2 with NU6140 inhibitor. <b>Chemical Biology & Drug Design</b>, v. 98, n. 5, p. 857–868, 2021.
</p>
<p align="justify">[3] CHELTSOV, Anton; NOMURA, Natsuko; YENUGONDA, Venkata; <i>et al</i>. Allosteric inhibitor of β-catenin selectively targets oncogenic Wnt signaling in colon cancer. <b>Scientific Reports</b>, v. 10, n. 1, p. 8096, 2020.</p>
</blockquote>
