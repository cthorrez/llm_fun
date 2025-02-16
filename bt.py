import polars as pl
import jax
import jax.numpy as jnp
from jax.scipy.optimize import minimize

def load_and_preprocess():
    df = pl.read_csv('data/collated_results.csv')
    models = [c for c in df.columns if c not in {'question', 'correct_letter', 'solve_rate'}]
    model_to_idx = {model:idx for idx, model in enumerate(models)}
    df = df.with_row_index(name='question_id').with_columns([
        pl.concat_list(models).alias('outcome'),
        pl.lit(list(range(len(models)))).cast(pl.List(pl.UInt32)).alias('model_id')
    ]).explode(['outcome', 'model_id'])
    matches = df[['model_id', 'question_id',]].to_jax()
    outcomes = df['outcome'].to_jax()
    return matches, outcomes


def loss_fn(
    matches,
    outcomes,
    model_ratings,
    question_ratings,
    model_reg,
    question_reg,
    ):
    rating_diffs = model_ratings[matches[:,0]] - question_ratings[matches[:,1]]
    probs = jax.nn.sigmoid(rating_diffs)
    jax.debug.print('probs {}', probs)
    ll = (jnp.log(probs) * outcomes).sum() + (jnp.log(1.0 - probs) * (1.0 - outcomes)).sum()
    reg = model_reg * jnp.linalg.norm(model_ratings) + question_reg * jnp.linalg.norm(question_ratings)
    loss = reg - ll
    jax.debug.print('loss {}', loss)
    return loss

loss_and_grad = jax.value_and_grad(
    fun=loss_fn,
    argnums=(2,3),
)


def bp_bt(matches, outcomes, n_models, n_questions, model_reg=1.0, question_reg=1.0):
    # initial_model_ratings = jnp.zeros(n_models)
    # initial_question_ratings = jnp.zeros(n_questions)
    key = jax.random.PRNGKey(0)
    initial_model_ratings = jax.random.normal(key=key, shape=(n_models,))
    initial_question_ratings = jax.random.normal(key=key, shape=(n_questions,))
    initial_params = jnp.concatenate([initial_model_ratings, initial_question_ratings])

    def objective(params):
        model_r = params[:n_models]
        question_r = params[n_models:]
        loss = loss_fn(
            matches, outcomes, model_r, question_r, model_reg, question_reg
        )
        return loss

    result = minimize(
        objective,
        initial_params,
        method='BFGS',
    )

    optimized_params = result.x
    model_ratings = optimized_params[:n_models]
    question_ratings = optimized_params[n_models:]
    return model_ratings, question_ratings

def main():
    matches, outcomes = load_and_preprocess()
    n_models = int(jnp.max(matches[:, 0])) + 1
    n_questions = int(jnp.max(matches[:, 1])) + 1
    model_ratings, question_ratings = bp_bt(matches, outcomes, n_models, n_questions)
    print("Model Ratings:", model_ratings)
    print("Question Ratings:", question_ratings)

if __name__ == '__main__':
    main()